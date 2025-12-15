# AI 챗봇 애플리케이션

FastAPI와 Jinja2를 사용한 챗봇 UI 프로젝트입니다.

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

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 서버 실행:
```bash
uvicorn app.main:app --reload
```

4. 브라우저에서 접속:
```
http://localhost:8000
```

## 다음 단계

- `app/routes/chat.py`에 AI 모델 연동 (OpenAI, Claude 등)
- 데이터베이스 추가 (대화 히스토리 저장)
- 사용자 인증 기능 추가
- 스트리밍 응답 구현

