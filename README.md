# Coding AI Decision Agent

개인 또는 팀의 예산·보안·IDE·업무 특성을 입력하면, 최신 정보 기반으로 코딩 AI 도구를 비교·평가·추천하는 LangGraph + Deep Research 기반 챗봇형 AI 에이전트입니다.

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
    ├─ HIT → 즉시 응답 (1초)
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
- 유사 질문: ~1초 (Redis 캐시)
- 관련 질문: ~5초 (Vector DB에서 Facts 재사용)

---

## 🔧 기술 스택

* **Backend**: FastAPI, LangGraph, LangChain
* **Python**: 3.11 (Docker), 3.12 (로컬 개발)
* **LLM**: OpenAI GPT-4o-mini
* **검색**: Tavily API / Serper.dev (Google Search)
* **캐싱**: Redis (최종 답변, 7일 TTL)
* **Vector DB**: Qdrant + Sentence Transformers (Facts 저장, 30일 TTL)
* **추적**: LangSmith

---

## 📁 프로젝트 구조

```
ai-agent/
├── app/                          # 메인 애플리케이션
│   ├── main.py                   # FastAPI 앱 진입점
│   ├── agent/                    # LangGraph 에이전트
│   │   ├── graph.py              # LangGraph 워크플로우 정의 (Main/Researcher/Supervisor Subgraph)
│   │   ├── nodes.py              # 에이전트 노드 함수들 (clarify, research, report 등)
│   │   ├── state.py              # LangGraph 상태 정의 (AgentState, ResearcherState, SupervisorState 등)
│   │   ├── prompts.py            # LLM 프롬프트 템플릿 (도메인 가이드, 리포트 생성, 쿼리 정규화 등)
│   │   ├── configuration.py      # 설정 관리 (모델, 토큰, 재시도 등)
│   │   ├── models.py             # Pydantic 모델 (ToolFact, UserContext, PricingPlan, SecurityPolicy 등)
│   │   ├── decision.py           # Decision Engine (질문 유형 판단: decision/comparison/explanation/information/guide)
│   │   ├── fact_extractor.py     # Facts 추출 유틸리티
│   │   ├── utils.py              # 공통 유틸리티 함수
│   │   ├── nodes/                # 노드 모듈 (clarifier, decision_maker, router, writer 등)
│   │   └── prompts/              # 프롬프트 모듈 (domain, clarify, research, report 등)
│   ├── tools/                    # 도구 모듈
│   │   ├── query_normalizer.py   # LLM 기반 쿼리 정규화 (의미적으로 동일한 질문 통합, 캐시 키 생성)
│   │   ├── cache.py              # Redis 캐시 관리 (최종 답변 저장/조회, TTL 관리, Fallback)
│   │   ├── vector_store.py       # Qdrant Vector DB 관리 (Facts 저장/검색, 임베딩 생성, 유사 질문 검색)
│   │   ├── search.py             # 웹 검색 도구 (Tavily API, Serper.dev Fallback, 검색 결과 Facts 추출)
│   │   └── __init__.py           # 모듈 초기화
│   ├── routes/                   # API 라우트
│   │   ├── chat.py               # 채팅 API 엔드포인트 (/api/chat, LangGraph 실행, 메시지 처리, 인사말/리포트 분리)
│   │   └── __init__.py           # 모듈 초기화
│   ├── static/                   # 정적 파일
│   │   ├── css/
│   │   │   └── style.css         # 스타일시트
│   │   └── js/
│   │       └── chat.js           # 프론트엔드 JavaScript (채팅 UI, 메시지 전송/수신)
│   └── templates/                # HTML 템플릿
│       └── index.html            # 채팅 인터페이스 (Jinja2 템플릿)
├── check_storage.py              # Redis/Vector DB 데이터 확인 스크립트 (통계 조회, 샘플 데이터 확인)
├── docker-compose.yml            # Docker Compose 설정 (app, redis, qdrant 서비스)
├── Dockerfile                    # 애플리케이션 Docker 이미지 (Python 3.11-slim, 의존성 설치, 모델 preload)
├── requirements.txt              # Python 의존성 (LangChain 1.0+, LangGraph 1.0+, Qdrant, Sentence Transformers 등)
├── env.example.txt               # 환경 변수 예시 (API 키, Redis, Qdrant 설정 등)
└── README.md                     # 프로젝트 문서
```

### 📂 주요 디렉토리 설명

#### `app/agent/` - LangGraph 에이전트
- **`graph.py`**: 전체 워크플로우 정의 (Main/Researcher/Supervisor Subgraph)
- **`nodes.py`**: 
  - `clarify_with_user`: 쿼리 정규화, 캐시 조회, 동적 인사말 생성
  - `researcher`: Vector DB 검색 → 정보 부족 시 웹 검색 (Tavily/Serper)
  - `final_report_generation`: 질문 유형별 리포트 생성, 캐시 저장
  - `run_decision_engine`: 질문 유형 판단 (decision/comparison/explanation/information/guide)
- **`state.py`**: LangGraph 상태 정의 (AgentState, ResearcherState, SupervisorState)
- **`prompts.py`**: LLM 프롬프트 템플릿 (도메인 가이드, 리포트 생성, 쿼리 정규화 등)
- **`nodes/`, `prompts/`**: 모듈화된 노드 및 프롬프트 (별도 디렉토리)

#### `app/tools/` - 도구 모듈
- **`query_normalizer.py`**: LLM 기반 쿼리 정규화, 캐시 키 생성 (`hash(normalized_text + keywords)`)
- **`cache.py`**: Redis 캐시 관리 (7일 TTL, Fallback: 메모리 캐시)
- **`vector_store.py`**: Qdrant Vector DB 관리, 유사 질문 검색 (30일 TTL)
- **`search.py`**: Tavily API 웹 검색, Serper.dev Fallback, Facts 추출

#### `app/routes/` - API 엔드포인트
- **`chat.py`**: `/api/chat` 엔드포인트, LangGraph 실행, 인사말/리포트 분리, 메시지 처리

---

## 🤖 에이전트 작동 방식

### LangGraph 워크플로우

```
clarify_with_user (쿼리 정규화 + 캐시 조회)
  ├─ 캐시 HIT → 즉시 응답 (인사말 + 리포트) → END
  └─ 캐시 MISS
      ↓
write_research_brief (연구 질문 작성, 질문 유형 분석)
      ↓
research_supervisor (연구 관리 및 할당)
      ↓
researcher (Vector DB 검색 → 정보 부족 시 웹 검색)
  ├─ Vector DB에서 충분한 정보 → LLM 답변 생성
  └─ 정보 부족 → Tavily/Serper 웹 검색 → Facts 저장 (30일 TTL)
      ↓
run_decision_engine (질문 유형 판단: decision/comparison/explanation/information/guide)
      ↓
final_report_generation (질문 유형별 리포트 생성, Redis 캐시 저장 7일 TTL) → END
```

**핵심 노드 설명**:
- **`clarify_with_user`**: Query Normalizer로 의미적으로 동일한 질문 통합 → Redis 캐시 조회 → HIT 시 동적 인사말 생성 후 즉시 응답
- **`researcher`**: Vector DB 우선 검색으로 웹 검색 최소화, 충분한 정보가 있으면 웹 검색 생략하여 비용 절감
- **`final_report_generation`**: 질문 유형에 따라 리포트 포맷팅 (추천 순위/비교/설명/정보/가이드), 인사말과 리포트 분리하여 2개 메시지 버블로 반환

---


## 🔄 변경 사항 (기존 코드 대비)

이 섹션은 기존 코드에서 발견된 버그 수정 및 부족했던 기능 보강 사항을 정리합니다.

### 🐛 버그 수정

1. **캐시 히트율 낮음**: Query Normalizer 추가 → 유사 질문 통합 → 응답 시간 99% 개선 (~10초 → ~0.1초)
2. **캐시 검증 부족**: 리포트 본문 길이 검증 (200자 이상) 추가
3. **인사말 없음**: `[GREETING]` 태그 지원, 동적 인사말 생성 로직 추가
4. **Follow-up 처리 부족**: 이전 추천 도구 추출/비교 로직 추가
5. **중복 답변**: 메시지 중복 제거 및 분리 로직 개선
6. **웹 검색 반복**: Vector DB 통합 → Facts 재사용 → 응답 시간 80% 개선 (~10초 → ~2초)
7. **캐시 키 해시 중복**: 해시 형태 검증 로직 수정

### 🔧 기능 보강

1. **하이브리드 캐싱 시스템**: Query Normalizer + Redis (7일 TTL) + Vector DB (30일 TTL)
2. **지능형 검색 전략**: Vector DB 우선 검색 → 정보 부족 시에만 웹 검색
3. **응답 포맷팅**: 2개 메시지 버블 분리 (인사말 + 리포트), 섹션 간 줄바꿈 개선
4. **Follow-up 컨텍스트**: 이전 추천 도구 추출/활용 로직 추가
5. **데이터 확인 도구**: `check_storage.py` 스크립트 추가
6. **환경 변수 관리**: Qdrant 설정 추가, Docker Compose 자동 설정

### 📦 라이브러리 업데이트

- **LangChain/LangGraph**: 0.x → 1.0+ (API 변경사항 반영)
- **Python**: 3.11 (Docker), 3.12 (로컬 개발)
