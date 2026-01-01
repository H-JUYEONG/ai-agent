"""Redis & Vector DB 저장 데이터 확인 스크립트"""

import json
from app.tools.vector_store import vector_store
from app.tools.cache import research_cache
from datetime import datetime


def print_separator(title: str):
    """구분선 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_vector_db():
    """Vector DB 통계 및 샘플 데이터 확인"""
    print_separator("Vector DB (Qdrant) 통계")
    
    stats = vector_store.get_stats()
    
    if not stats.get("available"):
        print("❌ Vector DB 비활성화됨")
        if "error" in stats:
            print(f"   오류: {stats['error']}")
        return
    
    print(f"✅ Vector DB 연결됨")
    print(f"   컬렉션: {stats.get('collection', 'N/A')}")
    print(f"   Points 개수: {stats.get('points_count', 0):,}개")
    print(f"   Vectors 개수: {stats.get('vectors_count', 0):,}개")
    print(f"   임베딩 차원: {stats.get('embedding_dim', 0)}차원")
    
    # 샘플 검색 (최근 저장된 데이터 확인)
    print_separator("Vector DB 샘플 검색")
    
    test_queries = [
        "Cursor 가격",
        "Copilot 보안",
        "AI 도구 비교"
    ]
    
    for query in test_queries:
        facts = vector_store.search_facts(query, limit=3, score_threshold=0.5)
        if facts:
            print(f"\n🔍 쿼리: '{query}'")
            print(f"   발견된 Facts: {len(facts)}개")
            for idx, fact in enumerate(facts[:2], 1):  # 최대 2개만 표시
                age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                print(f"\n   [{idx}] 신뢰도: {fact['score']:.3f} | {age_days:.1f}일 전")
                print(f"       내용: {fact['text'][:100]}...")
                print(f"       출처: {fact['source']} | {fact.get('url', '')[:50]}...")


def check_redis():
    """Redis 통계 및 샘플 데이터 확인"""
    print_separator("Redis 캐시 통계")
    
    stats = research_cache.get_stats()
    
    print(f"타입: {stats.get('type', 'N/A')}")
    print(f"캐시된 항목: {stats.get('cached_items', 0)}개")
    
    if stats.get("available"):
        print(f"✅ Redis 연결됨")
        print(f"   메모리 사용: {stats.get('memory_used', 'N/A')}")
    else:
        print(f"⚠️ Redis 비활성화 (메모리 캐시 사용 중)")
    
    # Redis에서 실제 키 확인 (Redis 사용 시)
    if research_cache.available and research_cache.redis:
        try:
            # 모든 ai-agent 키 찾기
            keys = research_cache.redis.keys("ai-agent:*")
            
            print_separator("Redis 저장된 키 목록")
            print(f"총 {len(keys)}개 키 발견")
            
            # 키 분류
            final_keys = [k for k in keys if ":final:" in k]
            search_keys = [k for k in keys if ":query:" in k or ":search:" in k]
            answer_keys = [k for k in keys if ":answer:" in k]
            
            print(f"\n📊 키 분류:")
            print(f"   최종 답변 (final): {len(final_keys)}개")
            print(f"   검색 결과 (query/search): {len(search_keys)}개")
            print(f"   일반 답변 (answer): {len(answer_keys)}개")
            
            # 샘플 키 조회 (최대 3개)
            if final_keys:
                print_separator("Redis 최종 답변 샘플 (최대 3개)")
                for key in final_keys[:3]:
                    try:
                        data = research_cache.redis.get(key)
                        if data:
                            value = json.loads(data)
                            ttl = research_cache.redis.ttl(key)
                            ttl_hours = ttl / 3600 if ttl > 0 else 0
                            
                            print(f"\n🔑 키: {key}")
                            print(f"   TTL: {ttl_hours:.1f}시간 남음")
                            content = value.get("content", value.get("reply", ""))
                            print(f"   내용: {content[:150]}...")
                    except Exception as e:
                        print(f"   ⚠️ 키 조회 실패: {e}")
            
        except Exception as e:
            print(f"⚠️ Redis 키 조회 실패: {e}")


def check_memory_cache():
    """메모리 캐시 확인 (Redis 없을 때)"""
    if not hasattr(research_cache, 'memory_cache'):
        return
    
    print_separator("메모리 캐시 통계")
    cache_size = len(research_cache.memory_cache)
    print(f"캐시된 항목: {cache_size}개")
    
    if cache_size > 0:
        print(f"\n샘플 키 (최대 5개):")
        for idx, key in enumerate(list(research_cache.memory_cache.keys())[:5], 1):
            print(f"   [{idx}] {key}")


def main():
    """메인 함수"""
    print("\n" + "🔍" * 30)
    print("  저장 데이터 확인 스크립트")
    print("🔍" * 30)
    
    # Vector DB 확인
    check_vector_db()
    
    # Redis 확인
    check_redis()
    
    # 메모리 캐시 확인
    if not research_cache.available:
        check_memory_cache()
    
    print_separator("확인 완료")
    print("\n💡 팁:")
    print("   - Vector DB: http://localhost:6333/dashboard")
    print("   - Redis CLI: docker exec -it ai-agent-redis redis-cli")
    print("   - 전체 삭제: vector_store.clear_all() 또는 research_cache.clear_all()")


if __name__ == "__main__":
    main()





