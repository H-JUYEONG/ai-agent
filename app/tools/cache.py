"""Redis 기반 연구 결과 캐싱"""

import hashlib
import json
import os
from typing import Optional, Dict, Any
import redis
from redis.exceptions import RedisError


class RedisCache:
    """Redis 기반 캐싱 (24시간 TTL)"""
    
    def __init__(self, ttl_hours: int = 24):
        self.ttl_seconds = ttl_hours * 3600
        
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
    
    def set(self, query: str, result: Dict[str, Any], domain: str = "general", prefix: str = "answer"):
        """캐시에 저장"""
        key = self._get_key(query, domain, prefix)
        
        if self.available and self.redis:
            try:
                self.redis.setex(
                    key,
                    self.ttl_seconds,
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
research_cache = RedisCache(ttl_hours=24)



