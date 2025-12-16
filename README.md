# AI Service Advisor - AI 서비스 비교 분석 챗봇

**LLM, 코딩 AI, 디자인 AI** 등 다양한 AI 서비스를 **최신 정보**로 **비교·분석**하는 Deep Research 챗봇입니다.

## ✨ 주요 기능

- 🎯 **도메인 특화**: LLM, 코딩 AI, 디자인 AI 3가지 분야 전문 분석
- 🔍 **Deep Research**: Open Deep Research 구조 기반 심층 연구
- ⚡ **스마트 캐싱**: 동일 질문 즉시 응답 (24시간 TTL)
- 🔄 **Fallback 검색**: Tavily → DuckDuckGo 자동 전환
- 📊 **상세 리포트**: 마크다운 형식 + 출처 포함
- 🚀 **병렬 처리**: 빠른 응답 (6~8초)

## 프로젝트 구조

```
ai-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 애플리케이션 메인
│   ├── routes/
│   │   ├── __init__.py
│   │   └── chat.py            # 챗봇 API 라우트
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css      # 스타일시트
│   │   └── js/
│   │       └── chat.js        # 챗봇 프론트엔드 로직
│   └── templates/
│       └── index.html         # 메인 페이지 템플릿
├── requirements.txt           # Python 패키지 의존성
└── README.md
```

## 설치 및 실행

1. Conda 가상환경 생성
```bash
conda create -n agent python=3.12
```

2. 가상환경 활성화
```bash
conda activate agent
```

3. 패키지 설치:
```bash
pip install -r requirements.txt
```

4. **Redis 설치 및 실행** (캐싱 최적화)

**Windows**:
```bash
# WSL2에서 Redis 실행
wsl
sudo apt update
sudo apt install redis-server
sudo service redis-server start
```

**Mac**:
```bash
brew install redis
brew services start redis
```

**Linux**:
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**Redis 연결 확인**:
```bash
redis-cli ping
# 응답: PONG
```

> **참고**: Redis 없이도 실행 가능 (자동으로 메모리 캐시로 대체됨)

5. 서버 실행
```bash
uvicorn app.main:app --reload
```

5. 브라우저에서 접속
```
http://localhost:8000
```

## 🏗️ 시스템 아키텍처

```
사용자 (도메인 선택: LLM/코딩/디자인)
    ↓
FastAPI (/api/chat)
    ↓
캐시 확인 (24시간 TTL)
    ├─ HIT → 즉시 응답 (0.1초)
    └─ MISS → Deep Research
        ↓
LangGraph 워크플로우
    ├─ clarify_with_user (범위 정제)
    ├─ write_research_brief (연구 계획)
    ├─ research_supervisor (반복 루프)
    │   ├─ supervisor (전략 수립)
    │   ├─ researcher × 3개 병렬
    │   │   ├─ Tavily 검색 (우선)
    │   │   └─ DuckDuckGo (fallback)
    │   └─ compress_research
    └─ final_report_generation
        ↓
캐시 저장 & 응답
```

## 🔧 기술 스택

- **Backend**: FastAPI, LangGraph, LangChain
- **LLM**: OpenAI (gpt-4o-mini, gpt-4o)
- **검색**: Tavily API (1차), DuckDuckGo (2차)
- **캐싱**: Redis (검색 쿼리 캐싱 + 최종 답변 캐싱, 24시간 TTL)
  - Fallback: 메모리 캐시 (Redis 없을 시 자동 대체)
- **추적**: LangSmith

## 📊 성능

- **첫 질문**: 4~6초 (검색 쿼리 캐싱으로 50% 단축)
- **재질문 (캐시)**: 0.01초 (Redis 기반)
- **유사 질문**: 2~3초 (일부 검색 재사용)
- **비용**: 리서치 1회당 $0.30~$0.60 (캐싱으로 50% 절감)
- **캐시 효율**: 
  - 동일 질문: 100% 재사용
  - 유사 질문: 60% 재사용

## 🔍 사용 예시

1. 도메인 선택: **LLM** 버튼 클릭
2. 질문: "GPT-4와 Claude Sonnet 비교해줘"
3. 결과: 상세한 마크다운 리포트 (출처 포함)

## 🛠️ 가상환경 관리

- **활성화**: `conda activate agent`
- **비활성화**: `conda deactivate`
- **삭제**: `conda remove -n agent --all`