"""Redis 기반 연구 결과 캐싱"""

import hashlib
import json
import os
from typing import Optional, Dict, Any
import redis
from redis.exceptions import RedisError


class RedisCache:
    """Redis 기반 캐싱 (검색 결과 + 최종 답변)"""
    
    def __init__(self, search_ttl_hours: int = 24, answer_ttl_hours: int = 168):
        """
        Args:
            search_ttl_hours: 검색 결과 TTL (기본 24시간)
            answer_ttl_hours: 최종 답변 TTL (기본 7일)
        """
        self.search_ttl_seconds = search_ttl_hours * 3600
        self.answer_ttl_seconds = answer_ttl_hours * 3600
        
        # Redis 연결 (localhost:6379 기본)
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # 연결 테스트
            self.redis.ping()
            self.available = True
            print(f"✅ Redis 연결 성공 ({redis_host}:{redis_port})")
        except (RedisError, ConnectionError) as e:
            self.redis = None
            self.available = False
            print(f"⚠️ Redis 연결 실패 - 메모리 캐시로 대체: {e}")
            # Fallback: 메모리 캐시
            self.memory_cache: Dict[str, Any] = {}
    
    def _get_key(self, query: str, domain: str, prefix: str = "answer") -> str:
        """캐시 키 생성"""
        # 이미 해시 형태(32자 hex)인 경우 그대로 사용
        if len(query) == 32 and all(c in '0123456789abcdef' for c in query.lower()):
            # 이미 해시인 경우
            return f"ai-agent:{prefix}:{domain}:{query}"
        # 일반 텍스트인 경우 해시 생성
        query_hash = hashlib.md5(f"{query.lower().strip()}".encode()).hexdigest()
        return f"ai-agent:{prefix}:{domain}:{query_hash}"
    
    def get(self, query: str, domain: str = "general", prefix: str = "answer") -> Optional[Dict[str, Any]]:
        """캐시에서 가져오기"""
        key = self._get_key(query, domain, prefix)
        
        if self.available and self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    print(f"✅ Redis 캐시 히트: {query[:50]}...")
                    return json.loads(data)
            except (RedisError, json.JSONDecodeError) as e:
                print(f"⚠️ Redis 읽기 오류: {e}")
        else:
            # Fallback: 메모리 캐시
            if key in self.memory_cache:
                print(f"✅ 메모리 캐시 히트: {query[:50]}...")
                return self.memory_cache[key]
        
        return None
    
    def set(self, query: str, result: Dict[str, Any], domain: str = "general", prefix: str = "answer", ttl_seconds: Optional[int] = None):
        """
        캐시에 저장
        
        Args:
            query: 쿼리 (또는 캐시 키)
            result: 저장할 데이터
            domain: 도메인
            prefix: 접두사 (answer / query / search)
            ttl_seconds: 사용자 정의 TTL (None이면 자동 선택)
        """
        key = self._get_key(query, domain, prefix)
        
        # TTL 자동 선택
        if ttl_seconds is None:
            if prefix in ["answer", "final"]:
                ttl_seconds = self.answer_ttl_seconds
            else:
                ttl_seconds = self.search_ttl_seconds
        
        if self.available and self.redis:
            try:
                self.redis.setex(
                    key,
                    ttl_seconds,
                    json.dumps(result, ensure_ascii=False)
                )
                # 통계 업데이트
                stats_key = f"ai-agent:stats:cache_count"
                self.redis.incr(stats_key)
                # 로그는 chat.py에서 출력하므로 여기서는 생략
            except (RedisError, TypeError) as e:
                print(f"⚠️ Redis 저장 오류: {e}")
        else:
            # Fallback: 메모리 캐시
            self.memory_cache[key] = result
            # 로그는 chat.py에서 출력하므로 여기서는 생략
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        if self.available and self.redis:
            try:
                cache_count = self.redis.get("ai-agent:stats:cache_count") or "0"
                info = self.redis.info("memory")
                return {
                    "type": "Redis",
                    "cached_items": cache_count,
                    "memory_used": info.get("used_memory_human", "N/A"),
                    "available": True
                }
            except RedisError:
                pass
        
        return {
            "type": "Memory",
            "cached_items": len(self.memory_cache),
            "available": False
        }
    
    def clear_all(self):
        """모든 캐시 삭제"""
        if self.available and self.redis:
            try:
                keys = self.redis.keys("ai-agent:*")
                if keys:
                    self.redis.delete(*keys)
                    print(f"🗑️ Redis 전체 캐시 삭제: {len(keys)}개")
            except RedisError as e:
                print(f"⚠️ Redis 삭제 오류: {e}")
        else:
            count = len(self.memory_cache)
            self.memory_cache.clear()
            print(f"🗑️ 메모리 전체 캐시 삭제: {count}개")


# 전역 캐시 인스턴스
research_cache = RedisCache(search_ttl_hours=24, answer_ttl_hours=168)  # 검색 24h, 답변 7일



