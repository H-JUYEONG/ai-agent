# Coding AI Decision Agent

팀 상황 기반 코딩 AI 도구 추천 에이전트

**설명**
팀의 예산·보안·IDE·업무 특성을 입력하면, 최신 정보 기반으로 코딩 AI 도구를 비교·평가·추천하는 LangGraph + Deep Research 기반 챗봇형 AI 에이전트입니다.

---

## 🔗 프로젝트 링크

* **데모 URL**: [https://chatboot.shop/](https://chatboot.shop/)
* **GitHub**: [https://github.com/H-JUYEONG/ai-agent](https://github.com/H-JUYEONG/ai-agent)

<p float="left">
  <img src="https://github.com/H-JUYEONG/ai-agent/raw/main/chatbot1.png" width="300" />
  <img src="https://github.com/H-JUYEONG/ai-agent/raw/main/chatbot2.png" width="300" />
  <img src="https://github.com/H-JUYEONG/ai-agent/raw/main/chatbot3.png" width="300" />
</p>

## ✨ 주요 기능

* 팀 상황 기반 맞춤 추천 (예산, 보안, IDE, 업무 특성)
* Deep Research 기반 심층 분석 및 순위 추천
* **🆕 하이브리드 캐싱 시스템**:
  * **Query Normalizer**: LLM 기반 쿼리 정규화 (의미적으로 동일한 질문 통합)
  * **Redis**: 최종 답변 캐싱 (7일 TTL) - 즉시 응답
  * **Qdrant Vector DB**: 검증된 사실(Facts) 저장 (30일 TTL) - 웹 검색 최소화
* **지능형 검색 전략**: Vector DB → 웹 검색 (Tavily/Serper) 순차 실행
* Fallback 검색: Tavily → Serper.dev 자동 전환
* 상세 리포트: 마크다운 형식 + 출처 포함

---

## 🏗️ 시스템 구조

```
사용자 질문
    ↓
① Query Normalizer (LLM 기반 쿼리 정규화)
    ↓
② Redis 최종 답변 캐시 조회
    ├─ HIT → 즉시 응답 (0.1초)
    └─ MISS
        ↓
③ Vector DB (Qdrant) Facts 검색
    ├─ 충분한 정보 있음 → LLM 답변 생성
    └─ 정보 부족
        ↓
④ 웹 검색 (Tavily/Serper)
    ↓
⑤ Vector DB에 Facts 저장 (TTL 30일)
    ↓
⑥ LLM 최종 답변 생성
    ↓
⑦ Redis 최종 답변 저장 (TTL 7일)
```

**성능 개선:**
- 첫 질문: ~10초 (웹 검색 필요)
- 유사 질문: ~0.1초 (Redis 캐시)
- 관련 질문: ~2초 (Vector DB에서 Facts 재사용)

---

## 🔧 기술 스택

* **Backend**: FastAPI, LangGraph, LangChain
* **LLM**: OpenAI GPT-4o-mini
* **검색**: Tavily API / Serper.dev (Google Search)
* **캐싱**: Redis (최종 답변, 7일 TTL)
* **Vector DB**: Qdrant + Sentence Transformers (Facts 저장, 30일 TTL)
* **추적**: LangSmith

---

## ⚡ 실행 방법

### 방법 1: Docker Compose (권장)

```bash
# 1. .env 파일 생성 (env.example.txt 참고)
cp env.example.txt .env
# API 키 입력: OPENAI_API_KEY, TAVILY_API_KEY, SERPER_API_KEY

# 2. Docker Compose 실행
docker-compose up -d

# 3. 브라우저에서 접속
http://localhost:8000
```

서비스 포함:
- **app**: FastAPI 애플리케이션 (포트 8000)
- **redis**: Redis 캐시 (포트 6379)
- **qdrant**: Qdrant Vector DB (포트 6333, 6334)

### 방법 2: 로컬 실행

1. **가상환경 생성 & 활성화**

```bash
conda create -n agent python=3.12
conda activate agent
```

2. **패키지 설치**

```bash
pip install -r requirements.txt
```

3. **Redis & Qdrant 설치 (선택)**

```bash
# Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

4. **서버 실행**

```bash
uvicorn app.main:app --reload
```

5. **브라우저에서 접속**: `http://localhost:8000`

**참고**: Redis/Qdrant 없이도 실행 가능 (메모리 캐시로 Fallback)
