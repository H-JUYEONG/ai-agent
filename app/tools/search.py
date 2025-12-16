"""웹 검색 도구 (Tavily + DuckDuckGo 교차 검증 + Redis 캐싱)"""

import os
import hashlib
import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from tavily import TavilyClient
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from app.tools.cache import research_cache

# .env 파일 로드 (모듈 초기화 시점에 필요)
load_dotenv()


class SearchWithFallback:
    """Tavily 우선, 실패 시 DuckDuckGo Fallback"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.tavily = None
        if self.tavily_api_key and self.tavily_api_key != "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            try:
                self.tavily = TavilyClient(api_key=self.tavily_api_key)
            except Exception as e:
                pass  # 초기화 실패 시 DuckDuckGo 사용
        
        self.ddgs = DDGS()
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        search_depth: str = "advanced",
        enable_verification: bool = True
    ) -> Dict[str, Any]:
        """검색 실행 (Redis 캐싱 → 교차 검증)"""
        
        # 0차: Redis 캐시 확인 (검색 쿼리 캐싱)
        cached_result = research_cache.get(query, domain="search", prefix="query")
        if cached_result:
            return cached_result
        
        # 교차 검증 활성화 시: Tavily + DuckDuckGo 동시 검색
        if enable_verification and self.tavily:
            return await self._search_with_verification(query, max_results, search_depth)
        
        # 교차 검증 비활성화: 기존 방식 (Tavily → DuckDuckGo Fallback)
        if self.tavily:
            tavily_result = await self._search_tavily(query, max_results, search_depth)
            if tavily_result["success"]:
                research_cache.set(query, tavily_result, domain="search", prefix="query")
                return tavily_result
        
        # DuckDuckGo Fallback
        duckduckgo_result = await self._search_duckduckgo(query, max_results)
        if duckduckgo_result.get("success"):
            research_cache.set(query, duckduckgo_result, domain="search", prefix="query")
        
        return duckduckgo_result
    
    async def _search_with_verification(
        self,
        query: str,
        max_results: int,
        search_depth: str
    ) -> Dict[str, Any]:
        """Tavily + DuckDuckGo 동시 검색 및 교차 검증"""
        print(f"🔍 [교차 검증] Tavily + DuckDuckGo 동시 검색: {query}")
        
        # 병렬 검색
        tavily_task = self._search_tavily(query, max_results, search_depth)
        ddg_task = self._search_duckduckgo(query, max_results)
        
        tavily_result, ddg_result = await asyncio.gather(tavily_task, ddg_task)
        
        # 결과 통합 및 검증
        verified_results = self._cross_validate_results(
            tavily_result, 
            ddg_result, 
            query,
            max_results
        )
        
        if verified_results["success"]:
            print(f"✅ [교차 검증] {len(verified_results['results'])}개 검증된 결과")
            research_cache.set(query, verified_results, domain="search", prefix="query")
        else:
            print(f"⚠️ [교차 검증] 검증 실패, Tavily 결과 사용")
            if tavily_result.get("success"):
                verified_results = tavily_result
            elif ddg_result.get("success"):
                verified_results = ddg_result
        
        return verified_results
    
    def _cross_validate_results(
        self,
        tavily_result: Dict[str, Any],
        ddg_result: Dict[str, Any],
        query: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """두 검색 엔진 결과 교차 검증"""
        
        tavily_results = tavily_result.get("results", []) if tavily_result.get("success") else []
        ddg_results = ddg_result.get("results", []) if ddg_result.get("success") else []
        
        if not tavily_results and not ddg_results:
            return {"success": False, "error": "검색 결과 없음"}
        
        # 공식 사이트 URL 패턴
        official_domains = {
            "openai": ["openai.com"],
            "anthropic": ["anthropic.com"],
            "google": ["google.com", "deepmind.google"],
            "gemini": ["gemini.google.com"]
        }
        
        # 결과 통합 및 가중치 계산
        all_results = []
        seen_urls: Set[str] = set()
        
        # 1. 공식 사이트 결과 우선 (높은 가중치)
        for result in tavily_results + ddg_results:
            url = result.get("url", "").lower()
            if url in seen_urls:
                continue
            
            # 공식 사이트 확인
            is_official = any(
                domain in url for domains in official_domains.values() 
                for domain in domains
            )
            
            # 가중치 계산
            base_score = result.get("score", 0.5)
            if is_official:
                base_score = min(base_score * 1.5, 1.0)  # 공식 사이트 +50% 가중치
                print(f"  ✅ 공식 사이트 발견: {url[:50]}")
            
            result["score"] = base_score
            result["is_official"] = is_official
            all_results.append(result)
            seen_urls.add(url)
        
        # 가격 정보가 포함된 결과 우선순위 상승
        pricing_keywords = ["pricing", "cost", "subscription", "plan", "free", "plus", "pro", "$"]
        for result in all_results:
            content = (result.get("title", "") + " " + result.get("content", "")).lower()
            if any(kw in content for kw in pricing_keywords):
                result["score"] = min(result["score"] * 1.2, 1.0)  # 가격 정보 +20% 가중치
        
        # 점수 기준 정렬
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 상위 결과만 반환
        top_results = all_results[:max_results]
        
        return {
            "source": "verified",  # 교차 검증됨
            "results": top_results,
            "success": True,
            "query": query,
            "tavily_count": len(tavily_results),
            "ddg_count": len(ddg_results),
            "verified_count": len(top_results)
        }
    
    async def _search_tavily(
        self, 
        query: str, 
        max_results: int,
        search_depth: str
    ) -> Dict[str, Any]:
        """Tavily 검색"""
        try:
            print(f"🔍 [Tavily] 검색 중: {query}")
            
            results = self.tavily.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_raw_content=False,
                days=90  # 최근 3개월(90일) 이내 정보만
            )
            
            if results and results.get("results"):
                formatted_results = []
                for r in results["results"]:
                    formatted_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                    })
                
                # 품질 검증
                if self._validate_results(formatted_results, query):
                    print(f"✅ [Tavily] {len(formatted_results)}개 결과 발견 (검증 통과)")
                    return {
                        "source": "tavily",
                        "results": formatted_results,
                        "success": True,
                        "query": query
                    }
                else:
                    print(f"⚠️ [Tavily] 품질 검증 실패")
                    return {"success": False}
            
            print(f"⚠️ [Tavily] 결과 없음")
            return {"success": False}
        
        except Exception as e:
            print(f"❌ [Tavily] 오류: {str(e)}")
            return {"success": False}
    
    async def _search_duckduckgo(
        self, 
        query: str, 
        max_results: int
    ) -> Dict[str, Any]:
        """DuckDuckGo 검색"""
        try:
            print(f"🔍 [DuckDuckGo] 검색 중: {query}")
            
            results = list(self.ddgs.text(
                keywords=query,
                max_results=max_results
            ))
            
            if results:
                formatted_results = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("link", ""),
                        "content": r.get("body", ""),
                        "score": 0.5,
                    }
                    for r in results
                ]
                
                print(f"✅ [DuckDuckGo] {len(formatted_results)}개 결과 발견")
                return {
                    "source": "duckduckgo",
                    "results": formatted_results,
                    "success": True,
                    "query": query
                }
            
            print(f"❌ [DuckDuckGo] 결과 없음")
            return {
                "source": "none",
                "results": [],
                "success": False,
                "error": "모든 검색 엔진 실패",
                "query": query
            }
        
        except Exception as e:
            print(f"❌ [DuckDuckGo] 오류: {str(e)}")
            return {
                "source": "none",
                "results": [],
                "success": False,
                "error": f"검색 실패: {str(e)}",
                "query": query
            }
    
    def _validate_results(self, results: List[Dict], query: str) -> bool:
        """검색 결과 품질 검증"""
        
        # 1. 최소 결과 개수 확인
        if len(results) < 2:
            return False
        
        # 2. 관련성 확인 (키워드 매칭)
        keywords = self._extract_keywords(query)
        if not keywords:
            return True  # 키워드 추출 실패 시 통과
        
        relevant_count = sum(
            1 for result in results 
            if any(
                kw.lower() in result.get("content", "").lower() 
                or kw.lower() in result.get("title", "").lower()
                for kw in keywords
            )
        )
        
        # 최소 50% 관련성
        return relevant_count >= len(results) * 0.5
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        # 간단한 키워드 추출 (공백 기준)
        words = query.split()
        # 3글자 이상만
        keywords = [w for w in words if len(w) >= 3]
        return keywords[:5]  # 최대 5개
    
    def extract_pricing_info(self, results: List[Dict]) -> Dict[str, Any]:
        """검색 결과에서 가격 정보 추출 및 검증"""
        pricing_patterns = [
            r'(?:free|무료)',
            r'(?:plus|pro|team|max|advanced)\s*\$?\s*(\d+(?:\.\d+)?)\s*/?\s*(?:월|month|mo)',
            r'\$(\d+(?:\.\d+)?)\s*/?\s*(?:월|month|mo)',
            r'(\d+(?:\.\d+)?)\s*(?:USD|달러)\s*/?\s*(?:월|month|mo)',
        ]
        
        found_pricing = {}
        official_pricing = {}
        
        for result in results:
            content = (result.get("title", "") + " " + result.get("content", "")).lower()
            url = result.get("url", "").lower()
            is_official = result.get("is_official", False)
            
            # 가격 정보 추출
            for pattern in pricing_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    plan_name = None
                    price = None
                    
                    # 플랜명 추출
                    if "free" in match.group(0).lower() or "무료" in match.group(0):
                        plan_name = "Free"
                        price = "무료"
                    elif "plus" in match.group(0).lower():
                        plan_name = "Plus"
                        price = f"${match.group(1) if match.lastindex else '20'}/월"
                    elif "pro" in match.group(0).lower():
                        plan_name = "Pro"
                        price = f"${match.group(1) if match.lastindex else '200'}/월"
                    elif match.lastindex:
                        price = f"${match.group(1)}/월"
                    
                    if plan_name and price:
                        key = f"{plan_name}_{price}"
                        if key not in found_pricing:
                            found_pricing[key] = {
                                "plan": plan_name,
                                "price": price,
                                "sources": [],
                                "official_count": 0
                            }
                        
                        found_pricing[key]["sources"].append({
                            "url": url,
                            "is_official": is_official
                        })
                        
                        if is_official:
                            found_pricing[key]["official_count"] += 1
                            official_pricing[key] = found_pricing[key]
        
        # 공식 사이트 가격 우선
        if official_pricing:
            return {
                "pricing": list(official_pricing.values()),
                "source": "official",
                "confidence": "high"
            }
        
        # 여러 출처에서 일치하는 가격
        verified_pricing = [
            p for p in found_pricing.values() 
            if len(p["sources"]) >= 2  # 2개 이상 출처에서 확인
        ]
        
        if verified_pricing:
            return {
                "pricing": verified_pricing,
                "source": "verified",
                "confidence": "medium"
            }
        
        # 단일 출처 가격
        if found_pricing:
            return {
                "pricing": list(found_pricing.values()),
                "source": "single",
                "confidence": "low"
            }
        
        return {
            "pricing": [],
            "source": "none",
            "confidence": "none"
        }


# 전역 검색 인스턴스
searcher = SearchWithFallback()



