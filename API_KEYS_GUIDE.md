# 🔑 API 키 발급 가이드

프로젝트 실행을 위해 3개의 API 키가 필요합니다.

## 1. OpenAI API Key (필수) ⭐

### 발급 방법:
1. https://platform.openai.com/ 접속
2. 회원가입 or 로그인
3. 좌측 메뉴 `API keys` 클릭
4. `Create new secret key` 버튼 클릭
5. 이름 입력 (예: ai-service-advisor)
6. 키 복사 (⚠️ 다시 볼 수 없으니 반드시 저장!)

### 형식:
```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 비용:
- gpt-4o-mini: $0.15 / 1M input tokens
- gpt-4o: $2.50 / 1M input tokens
- **예상**: 리서치 1회당 $0.50~$1.00
- **과제 50회**: 약 $25~50

### 무료 크레딧:
- 신규 가입 시 $5 무료

---

## 2. Tavily API Key (필수) ⭐

### 발급 방법:
1. https://tavily.com/ 접속
2. 회원가입 (이메일만 필요)
3. 자동으로 Dashboard로 이동
4. `API Keys` 탭에서 키 확인
5. 키 복사

### 형식:
```
tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 비용:
- **무료**: 1,000 requests/월 ✅
- Pro: $100/월 (10,000 requests)

### 무료로 충분:
- 과제 테스트 200회 가능

---

## 3. LangSmith API Key (필수) ⭐

### 발급 방법:
1. https://smith.langchain.com/ 접속
2. 회원가입 (Google 계정 가능)
3. Settings → `API Keys` 클릭
4. `Create API Key` 버튼 클릭
5. 이름 입력 (예: ai-service-advisor)
6. 키 복사

### 형식:
```
lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 비용:
- **무료**: Developer 플랜 ✅

### 용도:
- 그래프 흐름 시각화
- 각 노드 성능 측정
- 에러 추적
- 과제 발표 시 보여주기 좋음

---

## 📝 .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하세요:

```bash
# .env 파일 내용

# OpenAI API Key
OPENAI_API_KEY=sk-proj-여기에_실제_키_입력

# Tavily API Key
TAVILY_API_KEY=tvly-여기에_실제_키_입력

# LangSmith API Key
LANGSMITH_API_KEY=lsv2_pt_여기에_실제_키_입력
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-service-advisor

# Debug
DEBUG=false
```

**중요**: `여기에_실제_키_입력` 부분을 실제 발급받은 키로 교체하세요!

---

## ✅ 키 확인 방법

.env 파일 생성 후 서버 실행:

```bash
uvicorn app.main:app --reload
```

로그 확인:
```
✅ OpenAI API 키 확인 완료
✅ Tavily API 키 확인 완료
✅ LangSmith 클라이언트 초기화 성공
```

---

## 🚨 문제 해결

### "OpenAI API 키가 설정되지 않았습니다"
- .env 파일이 프로젝트 루트에 있는지 확인
- API 키 형식 확인 (sk-proj-로 시작)
- 따옴표 없이 입력했는지 확인

### "Tavily 검색 실패"
- API 키 형식 확인 (tvly-로 시작)
- 무료 한도(1,000회) 확인
- 자동으로 DuckDuckGo로 전환됨

### "LangSmith 초기화 실패"
- 디버깅용이므로 필수 아님
- LANGSMITH_TRACING=false로 설정 가능

---

## 💡 팁

1. **API 키 보안**
   - .gitignore에 .env 추가 (이미 포함됨)
   - GitHub에 절대 올리지 말 것

2. **비용 관리**
   - OpenAI Dashboard에서 사용량 모니터링
   - Usage limit 설정 권장 ($10~20)

3. **테스트**
   - 처음에는 간단한 질문으로 테스트
   - "GPT-4 가격은?" 같은 짧은 질문

---

## 4. Redis 설정 (선택 - 캐싱 최적화)

### ⚙️ Redis란?
- **검색 쿼리 캐싱**: 동일한 검색 쿼리를 재사용
- **50% 비용 절감**: 중복 검색 방지
- **영구 저장**: 서버 재시작해도 캐시 유지

### 📥 설치 방법

**Windows (WSL2)**:
```bash
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

### ✅ 연결 확인
```bash
redis-cli ping
# 응답: PONG
```

### 📝 .env 설정 (선택)
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

> **참고**: Redis 없이도 실행 가능! 자동으로 메모리 캐시로 대체됩니다.

---

**준비 완료!** 🎉

이제 `uvicorn app.main:app --reload`로 서버를 실행하세요!



