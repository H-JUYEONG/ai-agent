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
---

## ✨ 주요 기능

* 팀 상황 기반 맞춤 추천 (예산, 보안, IDE, 업무 특성)
* Deep Research 기반 심층 분석 및 순위 추천
* 스마트 캐싱: 동일 질문 즉시 응답 (24시간 TTL)
* Fallback 검색: Tavily → DuckDuckGo 자동 전환
* 상세 리포트: 마크다운 형식 + 출처 포함

---

## 🏗️ 시스템 구조

```
사용자 입력 → FastAPI → 캐시 확인
    ├─ HIT → 즉시 응답
    └─ MISS → LangGraph Deep Research
        ├─ clarify_with_user
        ├─ write_research_brief
        ├─ research_supervisor (Tavily / DuckDuckGo)
        └─ final_report_generation → 응답 + 캐시 저장
```

---

## 🔧 기술 스택

* **Backend**: FastAPI, LangGraph, LangChain
* **LLM**: OpenAI GPT-4o-mini
* **검색**: Tavily API / DuckDuckGo
* **캐싱**: Redis (Fallback: 메모리 캐시)
* **추적**: LangSmith

---

## ⚡ 실행 방법

1. 가상환경 생성 & 활성화

```bash
conda create -n agent python=3.12
conda activate agent
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. Redis 설치 (선택, 캐싱 최적화)
4. 서버 실행

```bash
uvicorn app.main:app --reload
```

5. 브라우저에서 접속: `http://localhost:8000`

---

## 📊 사용 예시

**입력**

```
예산: 50만원 이하
보안: 코드 외부 유출 금지
IDE: VS Code
업무 특성: 웹 개발
팀 규모: 5명
```

**출력**

* 팀 상황 기반 코딩 AI 도구 추천 순위
* 예산/보안/IDE/업무 적합성 평가
* 부적합 도구 제외 및 이유
* 상세 마크다운 리포트 (출처 포함)

