# Coding AI Decision Agent - 팀 상황 기반 코딩 AI 도입 의사결정 에이전트

팀의 **예산·보안·IDE·업무 특성**을 입력하면, **최신 정보 기반**으로 코딩 AI 도구를 **비교·평가·추천**해주는 LangGraph + Deep Research 기반 챗봇형 AI 에이전트입니다.

## ✨ 주요 기능

- 🎯 **팀 상황 기반 추천**: 예산, 보안, IDE, 업무 특성에 맞는 코딩 AI 도구 추천
- 🔍 **Deep Research**: Open Deep Research 구조 기반 심층 연구
- 💰 **예산 고려**: 사용자 예산 범위 내 도구만 추천
- 🔒 **보안 평가**: 코드 유출 방지 정책 등 보안 요구사항 반영
- 💻 **IDE 호환성**: 사용하는 IDE와 호환되는 도구만 추천
- ⚡ **스마트 캐싱**: 동일 질문 즉시 응답 (24시간 TTL)
- 🔄 **Fallback 검색**: Tavily → DuckDuckGo 자동 전환
- 📊 **상세 리포트**: 마크다운 형식 + 출처 포함 + 팀 상황 기반 추천 순위

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
사용자 (팀 상황 입력: 예산/보안/IDE/업무 특성)
    ↓
FastAPI (/api/chat)
    ↓
캐시 확인 (24시간 TTL)
    ├─ HIT → 즉시 응답 (0.1초)
    └─ MISS → Deep Research
        ↓
LangGraph 워크플로우
    ├─ clarify_with_user (팀 상황 파악: 예산/보안/IDE/업무 특성)
    ├─ write_research_brief (팀 상황 기반 연구 계획)
    ├─ research_supervisor (반복 루프)
    │   ├─ supervisor (전략 수립)
    │   ├─ researcher × 3개 병렬
    │   │   ├─ Tavily 검색 (우선)
    │   │   └─ DuckDuckGo (fallback)
    │   └─ compress_research
    └─ final_report_generation (팀 상황 기반 추천 리포트)
        ├─ 각 도구의 예산/보안/IDE/업무 적합성 평가
        ├─ 추천 순위 생성
        └─ 부적합 도구 제외 및 이유 명시
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

## 🎯 핵심 차별점

- ✅ **팀 상황 기반 맞춤 추천**: 단순 비교가 아닌 팀 상황에 맞는 도구 추천
- ✅ **다차원 평가**: 예산, 보안, IDE, 업무 특성을 종합적으로 고려
- ✅ **의사결정 지원**: 부적합한 도구 제외 및 명확한 추천 이유 제공
- ✅ **최신 정보**: 최근 3개월 이내 정보만 사용하여 정확한 가격 및 기능 정보 제공

## 🔍 사용 예시

### 입력 예시
```
예산: 월 50만원 이하
보안: 코드가 외부로 유출되면 안 됨
IDE: VS Code
업무 특성: 웹 개발 (React, Node.js)
팀 규모: 5명
```

### 출력 결과
- 팀 상황에 맞는 코딩 AI 도구 추천 순위
- 각 도구의 예산/보안/IDE/업무 적합성 평가
- 부적합한 도구 제외 및 이유 명시
- 상세한 마크다운 리포트 (출처 포함)

### 예시 질문
- "예산 50만원, VS Code 사용, 웹 개발하는데 추천해줘"
- "코드 유출 방지 중요, IntelliJ 사용, 모바일 앱 개발"
- "무료 플랜 있는 코딩 AI 추천해줘, VS Code 사용"

## 🛠️ 가상환경 관리

- **활성화**: `conda activate agent`
- **비활성화**: `conda deactivate`
- **삭제**: `conda remove -n agent --all`