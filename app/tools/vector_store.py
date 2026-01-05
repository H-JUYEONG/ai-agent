"""Qdrant 기반 벡터 스토어 - Facts 저장 및 검색"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
)
from sentence_transformers import SentenceTransformer
import hashlib


class VectorStore:
    """Qdrant 기반 벡터 스토어"""
    
    def __init__(
        self,
        collection_name: str = "ai_tool_facts",
        query_collection_name: str = "ai_tool_queries",  # 질문-캐시 키 매핑용
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """
        Args:
            collection_name: Qdrant 컬렉션 이름
            embedding_model: 임베딩 모델 이름
            qdrant_url: Qdrant 서버 URL (기본: localhost:6333)
            qdrant_api_key: Qdrant API 키 (클라우드 사용 시)
        """
        self.collection_name = collection_name
        self.query_collection_name = query_collection_name
        
        # Qdrant 클라이언트 초기화
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        try:
            if qdrant_api_key:
                # 클라우드 모드
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                # 로컬 모드
                self.client = QdrantClient(host=qdrant_url, port=qdrant_port)
            
            # 연결 테스트
            self.client.get_collections()
            self.available = True
            print(f"✅ Qdrant 연결 성공 ({qdrant_url}:{qdrant_port})")
        except Exception as e:
            self.client = None
            self.available = False
            print(f"⚠️ Qdrant 연결 실패 - Vector Store 비활성화: {e}")
            return
        
        # 임베딩 모델 로드
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ 임베딩 모델 로드 완료: {embedding_model} (dim={self.embedding_dim})")
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 실패: {e}")
            self.available = False
            return
        
        # 컬렉션 생성 (없으면)
        self._ensure_collection()
        self._ensure_query_collection()
    
    def _ensure_collection(self):
        """컬렉션이 없으면 생성"""
        if not self.available:
            return
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Qdrant 컬렉션 생성: {self.collection_name}")
            else:
                print(f"✅ Qdrant 컬렉션 존재: {self.collection_name}")
        except Exception as e:
            print(f"❌ 컬렉션 생성 실패: {e}")
            self.available = False
    
    def _ensure_query_collection(self):
        """질문-캐시 키 매핑 컬렉션 생성"""
        if not self.available:
            return
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.query_collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.query_collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Qdrant 질문 컬렉션 생성: {self.query_collection_name}")
            else:
                print(f"✅ Qdrant 질문 컬렉션 존재: {self.query_collection_name}")
        except Exception as e:
            print(f"❌ 질문 컬렉션 생성 실패: {e}")
    
    def _generate_id(self, text: str, source: str) -> str:
        """고유 ID 생성 (중복 방지)"""
        unique_str = f"{text}:{source}"
        return hashlib.md5(unique_str.encode()).hexdigest()
    
    def add_facts(
        self,
        facts: List[Dict[str, Any]],
        ttl_days: int = 30
    ) -> bool:
        """
        Facts를 Vector DB에 저장
        
        Args:
            facts: 저장할 facts 리스트
                   [{"text": "...", "source": "...", "metadata": {...}}, ...]
            ttl_days: 유효 기간 (일)
        
        Returns:
            성공 여부
        """
        if not self.available:
            return False
        
        try:
            points = []
            expire_timestamp = int((datetime.now() + timedelta(days=ttl_days)).timestamp())
            
            for fact in facts:
                text = fact.get("text", "")
                if not text:
                    continue
                
                # 임베딩 생성
                embedding = self.embedding_model.encode(text).tolist()
                
                # 메타데이터 구성
                payload = {
                    "text": text,
                    "source": fact.get("source", "unknown"),
                    "url": fact.get("url", ""),
                    "created_at": int(datetime.now().timestamp()),
                    "expire_at": expire_timestamp,
                    "metadata": fact.get("metadata", {})
                }
                
                # Point 생성
                point_id = self._generate_id(text, payload["source"])
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"✅ Vector DB 저장 완료: {len(points)}개 facts (TTL={ttl_days}일)")
                return True
            
            return False
        
        except Exception as e:
            print(f"❌ Vector DB 저장 실패: {e}")
            return False
    
    def search_facts(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        유사한 Facts 검색
        
        Args:
            query: 검색 쿼리
            limit: 최대 결과 개수
            score_threshold: 최소 유사도 (0~1)
        
        Returns:
            검색 결과 리스트 [{"text": "...", "score": 0.9, ...}, ...]
        """
        if not self.available:
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 현재 시간 (만료된 facts 제외)
            current_timestamp = int(datetime.now().timestamp())
            
            # 검색 (만료되지 않은 것만)
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="expire_at",
                            range=Range(gte=current_timestamp)
                        )
                    ]
                )
            )
            
            # 결과 포맷팅
            facts = []
            for result in results:
                facts.append({
                    "text": result.payload.get("text", ""),
                    "source": result.payload.get("source", ""),
                    "url": result.payload.get("url", ""),
                    "score": result.score,
                    "metadata": result.payload.get("metadata", {}),
                    "created_at": result.payload.get("created_at", 0)
                })
            
            if facts:
                print(f"✅ Vector DB 검색 완료: {len(facts)}개 facts (최고 점수: {facts[0]['score']:.3f})")
            else:
                print(f"⚠️ Vector DB 검색 결과 없음 (쿼리: {query[:50]}...)")
            
            return facts
        
        except Exception as e:
            print(f"❌ Vector DB 검색 실패: {e}")
            return []
    
    def delete_expired_facts(self) -> int:
        """만료된 facts 삭제"""
        if not self.available:
            return 0
        
        try:
            current_timestamp = int(datetime.now().timestamp())
            
            # 만료된 포인트 검색
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="expire_at",
                            range=Range(lt=current_timestamp)
                        )
                    ]
                ),
                limit=1000
            )
            
            expired_ids = [point.id for point in response[0]]
            
            if expired_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=expired_ids
                )
                print(f"🗑️ 만료된 facts 삭제: {len(expired_ids)}개")
                return len(expired_ids)
            
            return 0
        
        except Exception as e:
            print(f"❌ 만료 facts 삭제 실패: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Vector Store 통계"""
        if not self.available:
            return {"available": False}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "available": True,
                "collection": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def add_query_mapping(
        self,
        query: str,
        cache_key: str,
        normalized_text: str,
        domain: str = "general",
        ttl_days: int = 7
    ) -> bool:
        """
        질문-캐시 키 매핑 저장 (유사 질문 검색용)
        
        Args:
            query: 원본 질문
            cache_key: 캐시 키
            normalized_text: 정규화된 질문
            domain: 도메인
            ttl_days: 유효 기간 (일)
        
        Returns:
            성공 여부
        """
        if not self.available:
            return False
        
        try:
            # 질문 임베딩 생성
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 메타데이터 구성
            expire_timestamp = int((datetime.now() + timedelta(days=ttl_days)).timestamp())
            payload = {
                "query": query,
                "cache_key": cache_key,
                "normalized_text": normalized_text,
                "domain": domain,
                "created_at": int(datetime.now().timestamp()),
                "expire_at": expire_timestamp
            }
            
            # Point 생성
            point_id = self._generate_id(query, cache_key)
            point = PointStruct(
                id=point_id,
                vector=query_embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.query_collection_name,
                points=[point]
            )
            print(f"✅ 질문-캐시 키 매핑 저장: '{query[:50]}...' → {cache_key[:8]}...")
            return True
        
        except Exception as e:
            print(f"❌ 질문 매핑 저장 실패: {e}")
            return False
    
    def search_similar_query(
        self,
        query: str,
        domain: str = "general",
        limit: int = 1,
        score_threshold: float = 0.85  # 높은 유사도만 (85% 이상)
    ) -> Optional[Dict[str, Any]]:
        """
        유사한 질문 검색 (기존 캐시 키 찾기)
        
        Args:
            query: 검색할 질문
            domain: 도메인
            limit: 최대 결과 개수
            score_threshold: 최소 유사도 (0~1)
        
        Returns:
            {"query": "...", "cache_key": "...", "score": 0.9} 또는 None
        """
        if not self.available:
            return None
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 현재 시간 (만료된 매핑 제외)
            current_timestamp = int(datetime.now().timestamp())
            
            # 검색 (만료되지 않고 같은 도메인만)
            results = self.client.search(
                collection_name=self.query_collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="expire_at",
                            range=Range(gte=current_timestamp)
                        ),
                        FieldCondition(
                            key="domain",
                            match={"value": domain}
                        )
                    ]
                )
            )
            
            if results and len(results) > 0:
                result = results[0]
                print(f"✅ 유사 질문 발견: '{result.payload.get('query', '')[:50]}...' (유사도: {result.score:.3f})")
                return {
                    "query": result.payload.get("query", ""),
                    "cache_key": result.payload.get("cache_key", ""),
                    "normalized_text": result.payload.get("normalized_text", ""),
                    "score": result.score
                }
            
            return None
        
        except Exception as e:
            print(f"❌ 유사 질문 검색 실패: {e}")
            return None
    
    def clear_all(self):
        """모든 facts 삭제 (테스트용)"""
        if not self.available:
            return
        
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            print(f"🗑️ Vector Store 전체 삭제 완료")
        except Exception as e:
            print(f"❌ Vector Store 삭제 실패: {e}")


# 전역 벡터 스토어 인스턴스
vector_store = VectorStore()






