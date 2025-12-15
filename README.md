# AI Service Advisor - AI 서비스 비교 분석 챗봇

FastAPI + Jinja2로 구현한 AI 서비스 비교 분석 챗봇입니다.  
LLM, 코딩 AI, 디자인 AI 등 다양한 AI 서비스를 사용 목적에 맞게 비교하고 분석해줍니다.

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

1. Conda 가상환경 생성:
```bash
conda create -n ai-agent python=3.12
```

2. 가상환경 활성화:
```bash
conda activate ai-agent
```

3. 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 서버 실행:
```bash
uvicorn app.main:app --reload
```

5. 브라우저에서 접속:
```
http://localhost:8000
```

## 가상환경 관리

- **활성화**: `conda activate ai-agent`
- **비활성화**: `conda deactivate`
- **삭제**: `conda remove -n ai-agent --all`

## 다음 단계

- `app/routes/chat.py`에 AI 모델 연동 (LangGraph, LLM API 등)
- LLM, 코딩 AI, 디자인 AI 비교 분석 로직 구현
- 데이터베이스 추가 (대화 히스토리 저장)
- Deep Research 기능 연동

