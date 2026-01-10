"""웹 검색 도구 (Tavily + Serper.dev Fallback + Redis 캐싱)"""

import os
import hashlib
import re
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Set
from tavily import TavilyClient
from dotenv import load_dotenv
from app.tools.cache import research_cache

# .env 파일 로드 (모듈 초기화 시점에 필요)
load_dotenv()


class SearchWithFallback:
    """Tavily 우선, 실패 시 Serper.dev Fallback"""
    
    def __init__(self, tavily_api_key: Optional[str] = None, serper_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        
        self.tavily = None
        if self.tavily_api_key and self.tavily_api_key != "tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
            try:
                self.tavily = TavilyClient(api_key=self.tavily_api_key)
            except Exception as e:
                pass  # 초기화 실패 시 Serper 사용
    
    async def search(
        self, 
        query: str, 
        max_results: int = 5,
        search_depth: str = "advanced",
        enable_verification: bool = True
    ) -> Dict[str, Any]:
        """검색 실행 (Redis 캐싱 → 동적 깊이 조정 → 교차 검증)"""
        
        # 0차: Redis 캐시 확인 (검색 쿼리 캐싱)
        cached_result = research_cache.get(query, domain="search", prefix="query")
        if cached_result:
            return cached_result
        
        # 동적 깊이 조정: basic → intermediate → advanced
        # 첫 검색은 basic으로 빠르게 시작
        initial_depth = "basic"
        initial_max_results = min(max_results, 3)  # 첫 검색은 3개만
        
        # 교차 검증 활성화 시: Tavily + Serper Fallback
        if enable_verification and self.tavily:
            return await self._search_with_verification_dynamic(
                query, max_results, search_depth, initial_depth, initial_max_results
            )
        
        # 교차 검증 비활성화: 동적 깊이 조정
        if self.tavily:
            result = await self._search_tavily_dynamic(
                query, max_results, search_depth, initial_depth, initial_max_results
            )
            if result["success"]:
                research_cache.set(query, result, domain="search", prefix="query")
                return result
        
        # Serper Fallback
        serper_result = await self._search_serper(query, initial_max_results)
        if serper_result.get("success"):
            research_cache.set(query, serper_result, domain="search", prefix="query")
            return serper_result
        
        # 결과 부족 시 max_results 확장하여 재시도
        if initial_max_results < max_results:
            serper_result = await self._search_serper(query, max_results)
            if serper_result.get("success"):
                research_cache.set(query, serper_result, domain="search", prefix="query")
                return serper_result
        
        return serper_result
    
    async def _search_with_verification_dynamic(
        self,
        query: str,
        max_results: int,
        target_depth: str,
        initial_depth: str,
        initial_max_results: int
    ) -> Dict[str, Any]:
        """동적 깊이 조정: basic → intermediate → advanced"""
        print(f"🔍 [검색] Tavily 우선 검색 (동적 깊이): {query}")
        
        # 1단계: basic으로 빠르게 시도
        tavily_result = await self._search_tavily(query, initial_max_results, initial_depth)
        
        if tavily_result.get("success"):
            # 결과가 충분하면 바로 반환
            if len(tavily_result.get("results", [])) >= 2:
                print(f"✅ [Tavily] 충분한 결과 확보 (basic, {len(tavily_result['results'])}개)")
                research_cache.set(query, tavily_result, domain="search", prefix="query")
                return tavily_result
        
        # 2단계: 결과 부족 시 intermediate로 재시도
        if initial_depth == "basic" and target_depth in ["intermediate", "advanced"]:
            print(f"⚠️ [Tavily] basic 결과 부족, intermediate로 재시도...")
            tavily_result = await self._search_tavily(query, max_results, "intermediate")
            
            if tavily_result.get("success") and len(tavily_result.get("results", [])) >= 2:
                print(f"✅ [Tavily] 충분한 결과 확보 (intermediate, {len(tavily_result['results'])}개)")
                research_cache.set(query, tavily_result, domain="search", prefix="query")
                return tavily_result
        
        # 3단계: 여전히 부족하면 advanced로 재시도
        if target_depth == "advanced":
            print(f"⚠️ [Tavily] intermediate 결과 부족, advanced로 재시도...")
            tavily_result = await self._search_tavily(query, max_results, "advanced")
            
            if tavily_result.get("success"):
                print(f"✅ [Tavily] 결과 확보 (advanced, {len(tavily_result.get('results', []))}개)")
                research_cache.set(query, tavily_result, domain="search", prefix="query")
                return tavily_result
        
        # Tavily 실패 시에만 Serper 시도
        print(f"⚠️ [Tavily] 실패, Serper.dev 시도...")
        serper_result = await self._search_serper(query, max_results)
        
        if serper_result.get("success"):
            print(f"✅ [Serper] 결과 확보")
            research_cache.set(query, serper_result, domain="search", prefix="query")
            return serper_result
        
        # 둘 다 실패 시 다른 쿼리로 재시도 (최대 2번)
        print(f"⚠️ [검색 실패] Tavily와 Serper 모두 실패, 다른 쿼리로 재시도...")
        retry_queries = self._generate_retry_queries(query)
        
        for retry_query in retry_queries[:2]:  # 최대 2번만 재시도
            if retry_query == query:
                continue  # 원본 쿼리는 이미 시도했으므로 스킵
            
            print(f"🔄 [재시도] 쿼리 변형: {retry_query}")
            
            # Tavily로 재시도
            tavily_retry = await self._search_tavily(retry_query, max_results, "basic")
            if tavily_retry.get("success"):
                print(f"✅ [재시도 성공] Tavily: {retry_query}")
                research_cache.set(query, tavily_retry, domain="search", prefix="query")
                return tavily_retry
            
            # Serper로 재시도
            serper_retry = await self._search_serper(retry_query, max_results)
            if serper_retry.get("success"):
                print(f"✅ [재시도 성공] Serper: {retry_query}")
                research_cache.set(query, serper_retry, domain="search", prefix="query")
                return serper_retry
        
        # 모든 재시도 실패
        print(f"❌ [검색 실패] 모든 쿼리와 엔진 실패")
        return {
            "source": "none",
            "results": [],
            "success": False,
            "error": "모든 검색 엔진 및 재시도 실패",
            "query": query
        }
    
    async def _search_with_verification(
        self,
        query: str,
        max_results: int,
        search_depth: str
    ) -> Dict[str, Any]:
        """기존 메서드 (하위 호환성)"""
        return await self._search_with_verification_dynamic(
            query, max_results, search_depth, "basic", min(max_results, 3)
        )
    
    async def _search_tavily_dynamic(
        self,
        query: str,
        max_results: int,
        target_depth: str,
        initial_depth: str,
        initial_max_results: int
    ) -> Dict[str, Any]:
        """동적 깊이 조정: basic → intermediate → advanced"""
        # 1단계: basic으로 빠르게 시도
        result = await self._search_tavily(query, initial_max_results, initial_depth)
        
        if result.get("success") and len(result.get("results", [])) >= 2:
            return result
        
        # 2단계: 결과 부족 시 intermediate로 재시도
        if initial_depth == "basic" and target_depth in ["intermediate", "advanced"]:
            result = await self._search_tavily(query, max_results, "intermediate")
            if result.get("success") and len(result.get("results", [])) >= 2:
                return result
        
        # 3단계: 여전히 부족하면 advanced로 재시도
        if target_depth == "advanced":
            result = await self._search_tavily(query, max_results, "advanced")
            if result.get("success"):
                return result
        
        return result
    
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
        """Tavily 검색 (타임아웃 적용)"""
        try:
            print(f"🔍 [Tavily] 검색 중 ({search_depth}): {query}")
            
            # site: 검색에서 오류 발생 시 일반 검색으로 대체
            original_query = query
            if "site:" in query.lower():
                # site: 검색을 시도하되, 오류 발생 시 일반 검색으로 대체
                pass
            
            # 타임아웃 설정: basic/intermediate는 8초, advanced는 12초
            timeout = 8.0 if search_depth in ["basic", "intermediate"] else 12.0
            
            # 동기 함수를 비동기로 실행 (타임아웃 적용)
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.tavily.search,
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth,
                    include_raw_content=False,
                    days=90
                ),
                timeout=timeout
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
        
        except asyncio.TimeoutError:
            print(f"⏱️ [Tavily] 타임아웃 ({search_depth})")
            return {"success": False, "timeout": True}
        
        except Exception as e:
            error_str = str(e)
            # 400, 432 오류는 site: 검색에서 자주 발생
            # Tavily는 site: 검색을 잘 지원하지 않으므로, 실패 시 바로 Serper.dev로 넘어감
            # (일반 검색으로 대체하면 특정 사이트 검색 의도가 사라지므로 의미 없음)
            if ("site:" in query.lower()) and ("400" in error_str or "432" in error_str or "Bad Request" in error_str):
                print(f"⚠️ [Tavily] site: 검색 오류 ({error_str[:50]}), Tavily는 site: 검색을 지원하지 않습니다. Serper.dev로 전환됩니다.")
            print(f"❌ [Tavily] 오류: {error_str}")
            return {"success": False}
    
    def _generate_retry_queries(self, original_query: str) -> List[str]:
        """검색 실패 시 재시도할 쿼리 목록 생성"""
        retry_queries = []
        
        # 1. site: 제거하고 일반 검색
        if "site:" in original_query.lower():
            general_query = re.sub(r'site:\S+\s*', '', original_query, flags=re.IGNORECASE).strip()
            if general_query and general_query != original_query:
                retry_queries.append(general_query)
        
        # 2. 쿼리를 단순화 (특수 문자 제거, 키워드만 추출)
        # 핵심 키워드 추출 (도구명, 핵심 개념)
        keywords = re.findall(r'\b[A-Z][a-zA-Z]+\b', original_query)  # 대문자로 시작하는 단어
        keywords.extend(re.findall(r'\b\w+\b', original_query.lower()))  # 모든 단어
        
        # 중복 제거 및 길이 기준 정리
        keywords = [kw for kw in set(keywords) if len(kw) > 3 and kw.lower() not in ['site', 'pricing', 'features', 'integration']]
        
        if keywords:
            # 핵심 키워드만으로 쿼리 생성 (최대 5개)
            simplified = ' '.join(keywords[:5])
            if simplified and simplified != original_query.lower():
                retry_queries.append(simplified)
            
            # 도구명 + "pricing" 또는 "features" 조합
            tool_names = [kw for kw in keywords[:3] if kw[0].isupper()]
            if tool_names:
                for suffix in ['pricing', 'features', 'review']:
                    tool_query = f"{' '.join(tool_names)} {suffix}"
                    if tool_query != original_query.lower():
                        retry_queries.append(tool_query)
        
        # 3. 원본 쿼리의 연도 제거 (예: "2026" 제거)
        year_removed = re.sub(r'\b20\d{2}\b', '', original_query).strip()
        if year_removed and year_removed != original_query:
            retry_queries.append(year_removed)
        
        return retry_queries
    
    async def _search_serper(
        self, 
        query: str, 
        max_results: int
    ) -> Dict[str, Any]:
        """Serper.dev 검색 (Google 검색 결과 제공, 타임아웃 5초)"""
        
        if not self.serper_api_key:
            print(f"❌ [Serper] API 키 없음")
            return {
                "source": "none",
                "results": [],
                "success": False,
                "error": "Serper API 키 없음",
                "query": query
            }
        
        try:
            print(f"🔍 [Serper] 검색 중: {query}")
            
            url = "https://google.serper.dev/search"
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            payload = {
                'q': query,
                'num': max_results
            }
            
            # aiohttp로 비동기 요청 (타임아웃 5초)
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # organic 결과 파싱
                        organic_results = data.get('organic', [])
                        
                        if organic_results:
                            formatted_results = [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("link", ""),
                                    "content": r.get("snippet", ""),
                                    "score": 0.8,  # Serper는 Google 검색이라 높은 점수
                                }
                                for r in organic_results[:max_results]
                            ]
                            
                            print(f"✅ [Serper] {len(formatted_results)}개 결과 발견")
                            return {
                                "source": "serper",
                                "results": formatted_results,
                                "success": True,
                                "query": query
                            }
                        
                        print(f"❌ [Serper] 결과 없음")
                        return {
                            "source": "none",
                            "results": [],
                            "success": False,
                            "error": "검색 결과 없음",
                            "query": query
                        }
                    
                    else:
                        error_text = await response.text()
                        print(f"❌ [Serper] HTTP {response.status}: {error_text}")
                        return {
                            "source": "none",
                            "results": [],
                            "success": False,
                            "error": f"Serper API 오류: {response.status}",
                            "query": query
                        }
        
        except asyncio.TimeoutError:
            print(f"❌ [Serper] 타임아웃")
            return {
                "source": "none",
                "results": [],
                "success": False,
                "error": "Serper 타임아웃",
                "query": query
            }
        
        except Exception as e:
            print(f"❌ [Serper] 오류: {str(e)}")
            return {
                "source": "none",
                "results": [],
                "success": False,
                "error": f"Serper 검색 실패: {str(e)}",
                "query": query
            }
    
    def _validate_results(self, results: List[Dict], query: str) -> bool:
        """검색 결과 품질 검증 (완화된 기준)"""
        
        # 1. 최소 결과 개수 확인 (완화: 2개 → 1개)
        if len(results) < 1:
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
        
        # 관련성 기준 완화 (50% → 30%)
        return relevant_count >= len(results) * 0.3
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 키워드 추출"""
        # 간단한 키워드 추출 (공백 기준)
        words = query.split()
        # 3글자 이상만
        keywords = [w for w in words if len(w) >= 3]
        return keywords[:5]  # 최대 5개
    
    def extract_pricing_info(self, results: List[Dict]) -> Dict[str, Any]:
        """검색 결과에서 가격 정보 추출 및 검증
        
        주의: 이 함수는 가격 정보만 추출합니다. 
        - 플랜명은 검색 결과에서 실제로 나온 것을 그대로 사용 (하드코딩 금지)
        - 개인용/비즈니스용 구별은 에이전트(LLM)가 검색 결과의 전체 컨텍스트를 보고 판단해야 합니다.
        """
        
        found_pricing = {}
        official_pricing = {}
        
        for result in results:
            # 원본 텍스트 유지 (대소문자 구별)
            title = result.get("title", "")
            content = result.get("content", "")
            full_text = f"{title} {content}"
            url = result.get("url", "").lower()
            is_official = result.get("is_official", False)
            
            # 가격 패턴 (플랜명은 검색 결과에서 실제로 나온 것을 추출)
            # 주의: 하드코딩된 플랜명(Free, Pro, Plus 등) 사용 금지
            pricing_patterns = [
                # 플랜명과 가격 함께 (예: "Pro $10/월", "Pro+ $20/월", "Business $19/월")
                # 플랜명은 도구마다 다르므로 하드코딩하지 않고 실제 검색 결과에서 추출
                (r'(?:^|\s)([A-Za-z가-힣][A-Za-z0-9가-힣\s\+]*?)\s*\$?\s*(\d+(?:\.\d+)?)\s*/?\s*(?:월|month|mo)', None, None),
                # 일반 가격 (예: "$10/월", "10 USD/월")
                (r'\$(\d+(?:\.\d+)?)\s*/?\s*(?:월|month|mo)', None, None),
                (r'(\d+(?:\.\d+)?)\s*(?:USD|달러)\s*/?\s*(?:월|month|mo)', None, None),
                # Free/무료 (특별 처리)
                (r'(?:^|\s)(?:free|무료)(?:\s|$)', None, "무료"),
            ]
            
            # 가격 정보 추출
            for pattern, default_plan, default_price in pricing_patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    plan_name = default_plan
                    price = default_price
                    
                    # 플랜명 및 가격 추출
                    groups = match.groups()
                    if default_price is None and len(groups) >= 2:
                        # 플랜명과 가격 모두 있는 경우
                        plan_name_match = groups[0].strip()
                        price_match = groups[1]
                        if plan_name_match and price_match:
                            # 플랜명은 검색 결과에서 나온 그대로 사용 (하드코딩 금지)
                            plan_name = plan_name_match
                            price = f"${price_match}/월"
                    elif default_price is None and len(groups) >= 1:
                        # 가격만 있는 경우
                        price_match = groups[0]
                        if price_match:
                            price = f"${price_match}/월"
                            plan_name = None  # 플랜명 없음
                    elif default_price:
                        # Free/무료인 경우
                        price = default_price
                        plan_name = "Free"  # Free는 일반적으로 사용되지만, 실제 검색 결과에서 확인 필요
                    
                    # 가격 정보 저장 (플랜명이 있으면 포함, 없으면 가격만)
                    if price:
                        # 키 생성 (플랜명이 있으면 포함)
                        if plan_name:
                            key = f"{plan_name}_{price}"
                        else:
                            key = f"unknown_{price}"
                        
                        if key not in found_pricing:
                            found_pricing[key] = {
                                "plan": plan_name,  # 플랜명이 없으면 None
                                "price": price,
                                "sources": [],
                                "official_count": 0,
                                "context": full_text[:300]  # 컨텍스트 보존 (에이전트가 개인용/비즈니스용 판단할 수 있도록)
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



