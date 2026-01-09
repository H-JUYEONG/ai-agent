# 🚀 Vector DB 통합 가이드

## 📊 개요

이 프로젝트는 **Hybrid 캐싱 시스템**을 도입하여 답변 속도를 10배 이상 개선했습니다.

```
기존: 매번 웹 검색 (~10초)
개선: Redis/Vector DB 활용 (0.1~2초)
```

---

## 🎯 3단계 캐싱 전략

### 1️⃣ Query Normalizer (LLM 기반)

**목적**: 의미적으로 동일한 질문을 같은 캐시 키로 변환

**예시**:
```
"Copilot 회사에서 써도 괜찮아요?"
"깃헙 코파일럿 보안 문제 없나요?"
→ 정규화: "Copilot 기업 사용 가능 여부"
→ 캐시 키: md5("Copilot 기업 사용 가능 여부:Copilot:기업:보안")
```

**구현**:
- `app/tools/query_normalizer.py`
- LLM (GPT-4o-mini) 사용
- 키워드 추출 + 의도 파악

---

### 2️⃣ Redis (최종 답변 캐싱)

**목적**: 완전히 동일한 질문에 대한 즉시 응답

**특징**:
- TTL: 7일 (168시간)
- Prefix: `final`
- Key: `ai-agent:final:{domain}:{cache_key}`

**성능**:
- 캐시 HIT: ~0.1초
- 정규화된 쿼리 기반이라 유사 질문도 HIT

**구현**:
```python
# app/agent/nodes.py - clarify_with_user()
cached_answer = research_cache.get(cache_key, domain=domain, prefix="final")
if cached_answer:
    return AIMessage(content=cached_answer["content"])  # 즉시 반환
```

---

### 3️⃣ Qdrant Vector DB (Facts 저장)

**목적**: 검증된 사실(Facts)을 저장하여 웹 검색 최소화

**특징**:
- TTL: 30일
- 임베딩: Sentence Transformers (all-MiniLM-L6-v2)
- 의미 기반 검색 (Cosine Similarity ≥ 0.75)

**저장 데이터**:
```json
{
  "text": "Cursor Pro: $20/월, 500 completions/월",
  "source": "tavily",
  "url": "https://cursor.com/pricing",
  "score": 0.95,
  "created_at": 1735545600,
  "expire_at": 1738137600,
  "metadata": {
    "is_official": true,
    "query": "Cursor pricing"
  }
}
```

**성능**:
- Vector DB HIT: ~2초 (웹 검색 스킵)
- 관련 질문에도 Facts 재사용 가능

**구현**:
```python
# app/agent/nodes.py - researcher()
async def vector_search(query: str) -> str:
    facts = vector_store.search_facts(query, limit=5, score_threshold=0.75)
    # ...

async def web_search(query: str) -> str:
    result = await searcher.search(...)
    # 웹 검색 후 Vector DB에 저장
    vector_store.add_facts(facts_to_store, ttl_days=30)
```

---

## 🔄 전체 워크플로우

```
사용자 질문: "Cursor 가격이 어떻게 되나요?"
    ↓
① Query Normalizer
   → "Cursor 가격 정보" (cache_key: abc123...)
    ↓
② Redis 최종 답변 조회
   → MISS
    ↓
③ Vector DB Facts 검색
   Query: "Cursor 가격 정보"
   → HIT: 3개 Facts 발견
      - "Cursor Pro: $20/월, 500 completions" (신뢰도 0.92, 5일 전)
      - "Cursor Business: $40/월, unlimited" (신뢰도 0.88, 5일 전)
      - "Cursor Free: 2000 completions/월" (신뢰도 0.85, 10일 전)
    ↓
④ LLM 답변 생성
   (웹 검색 스킵! Vector DB의 Facts만 사용)
    ↓
⑤ Redis 최종 답변 저장
   Key: "ai-agent:final:AI 서비스:abc123..."
   TTL: 7일
    ↓
응답 반환 (~2초)
```

**유사 질문 처리**:
```
다음 질문: "커서 얼마에요?"
    ↓
① Query Normalizer
   → "Cursor 가격 정보" (동일한 cache_key!)
    ↓
② Redis 최종 답변 조회
   → HIT! (정규화 덕분에 동일한 키)
    ↓
응답 반환 (~0.1초)
```

---

## 🛠️ 설정

### 환경 변수 (.env)

```bash
# Qdrant 설정
QDRANT_URL=localhost
QDRANT_PORT=6333
# QDRANT_API_KEY=your_key  # 클라우드 사용 시만

# Redis 설정
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### Docker Compose

```bash
# 전체 실행 (Redis + Qdrant 포함)
docker-compose up -d

# 개별 실행
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 임베딩 모델 다운로드 (첫 실행 시 자동)
# sentence-transformers/all-MiniLM-L6-v2 (~80MB)
```

---

## 📈 성능 비교

| 시나리오 | 기존 | 개선 | 개선율 |
|---------|------|------|--------|
| 첫 질문 | ~10초 | ~10초 | 0% (웹 검색 필요) |
| 동일 질문 반복 | ~10초 | ~0.1초 | **100배** |
| 유사 질문 | ~10초 | ~0.1초 | **100배** (정규화) |
| 관련 질문 (다른 측면) | ~10초 | ~2초 | **5배** (Vector DB) |

**예시**:
```
Q1: "Cursor 가격이 어떻게 되나요?" → 10초 (웹 검색)
Q2: "커서 얼마에요?" → 0.1초 (Redis, 정규화 덕분)
Q3: "Cursor 무료 버전 있어?" → 2초 (Vector DB, 가격 Facts 재사용)
Q4: "Cursor Business 기능은?" → 2초 (Vector DB, Business Facts 재사용)
```

---

## 🔍 모니터링

### Vector DB 통계

```python
from app.tools.vector_store import vector_store

stats = vector_store.get_stats()
print(stats)
# {
#   "available": True,
#   "collection": "ai_tool_facts",
#   "points_count": 1234,
#   "vectors_count": 1234,
#   "embedding_dim": 384
# }
```

### 만료된 Facts 삭제

```python
deleted_count = vector_store.delete_expired_facts()
print(f"삭제된 Facts: {deleted_count}개")
```

### Redis 캐시 통계

```python
from app.tools.cache import research_cache

stats = research_cache.get_stats()
print(stats)
# {
#   "type": "Redis",
#   "cached_items": "567",
#   "memory_used": "2.3M",
#   "available": True
# }
```

---

## 🐛 트러블슈팅

### Qdrant 연결 실패

```
⚠️ Qdrant 연결 실패 - Vector Store 비활성화
```

**해결책**:
1. Qdrant가 실행 중인지 확인
   ```bash
   docker ps | grep qdrant
   ```

2. 포트 확인
   ```bash
   curl http://localhost:6333/healthz
   ```

3. 환경 변수 확인
   ```bash
   echo $QDRANT_URL
   echo $QDRANT_PORT
   ```

**참고**: Qdrant 없이도 실행 가능 (웹 검색만 사용)

### 임베딩 모델 로드 실패

```
❌ 임베딩 모델 로드 실패
```

**해결책**:
1. 디스크 공간 확인 (모델 크기: ~80MB)
2. 인터넷 연결 확인 (첫 실행 시 다운로드)
3. 수동 다운로드:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   ```

### Redis 캐시 미작동

```
⚠️ Redis 연결 실패 - 메모리 캐시로 대체
```

**해결책**:
- Redis 없이도 실행 가능 (메모리 캐시 자동 전환)
- 프로덕션 환경에서는 Redis 사용 권장

---

## 🎓 Best Practices

### TTL 설정 전략

| 데이터 종류 | TTL | 이유 |
|-------------|-----|------|
| 최종 답변 (Redis) | 7일 | 자주 변경되는 정보 (가격, 기능 업데이트) |
| Facts (Vector DB) | 30일 | 사실 정보는 상대적으로 안정적 |
| 검색 결과 (Redis) | 24시간 | 웹 검색 결과는 빠르게 변경 가능 |

### 임베딩 모델 선택

현재: `sentence-transformers/all-MiniLM-L6-v2`
- 장점: 빠름 (~50ms), 작음 (~80MB), 무료
- 단점: 정확도는 대형 모델보다 낮음

대안:
```python
# 고정밀 모델 (느리지만 정확)
"sentence-transformers/all-mpnet-base-v2"  # 420MB, ~100ms

# OpenAI Embeddings (유료)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### 검색 임계값 (score_threshold)

```python
# 엄격한 검색 (정확도 우선)
vector_store.search_facts(query, score_threshold=0.85)

# 느슨한 검색 (재현율 우선)
vector_store.search_facts(query, score_threshold=0.65)

# 권장값
vector_store.search_facts(query, score_threshold=0.75)  # 균형
```

---

## 📚 참고 자료

- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [Redis 캐싱 전략](https://redis.io/docs/manual/patterns/caching/)
- [Vector DB 비교](https://benchmark.vectorview.ai/)

---

## 🤝 기여

개선 아이디어:
1. 임베딩 모델 업그레이드 (더 정확한 검색)
2. 하이브리드 검색 (키워드 + 벡터)
3. 캐시 워밍 (인기 질문 미리 저장)
4. A/B 테스트 (캐시 전략 비교)









