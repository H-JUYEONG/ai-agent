# GitHub Actions CI/CD 설정 가이드 🚀

## 📋 개요

GitHub에 코드를 푸시하면 자동으로 AWS EC2에 배포됩니다.

```
코드 푸시 (main 브랜치)
    ↓
GitHub Actions 트리거
    ↓
Docker 이미지 빌드
    ↓
EC2에 배포
    ↓
자동 재시작
```

---

## 1️⃣ EC2 초기 설정 (최초 1회만)

### 1-1. SSH 접속
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

### 1-2. Docker 설치
```bash
# Docker 설치
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 1-3. 프로젝트 디렉토리 생성
```bash
# 재접속 (docker 그룹 적용)
exit
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 프로젝트 디렉토리 생성
mkdir -p ~/ai-agent
cd ~/ai-agent
```

---

## 2️⃣ GitHub Secrets 설정

### 2-1. GitHub 저장소로 이동
```
https://github.com/YOUR_USERNAME/YOUR_REPO
```

### 2-2. Settings → Secrets and variables → Actions

**"New repository secret" 클릭하여 다음 항목들을 추가:**

---

### 📝 추가할 Secrets 목록

#### 1. **EC2_HOST**
- **Name:** `EC2_HOST`
- **Value:** `YOUR_EC2_PUBLIC_IP`
- **예시:** `3.35.123.45`

#### 2. **EC2_USERNAME**
- **Name:** `EC2_USERNAME`
- **Value:** `ubuntu`

#### 3. **EC2_SSH_KEY**
- **Name:** `EC2_SSH_KEY`
- **Value:** EC2 키 페어 내용 (전체)

**키 파일 내용 복사 방법:**
```bash
# Windows PowerShell
Get-Content your-key.pem | clip

# 또는 메모장으로 열어서 전체 복사
# -----BEGIN RSA PRIVATE KEY-----
# (전체 내용)
# -----END RSA PRIVATE KEY-----
```

**⚠️ 중요:**
- 첫 줄 `-----BEGIN RSA PRIVATE KEY-----`부터
- 마지막 줄 `-----END RSA PRIVATE KEY-----`까지
- **전체를 복사**하세요!

#### 4. **OPENAI_API_KEY**
- **Name:** `OPENAI_API_KEY`
- **Value:** `your_openai_api_key`

#### 5. **TAVILY_API_KEY**
- **Name:** `TAVILY_API_KEY`
- **Value:** `your_tavily_api_key`

#### 6. **SERPER_API_KEY**
- **Name:** `SERPER_API_KEY`
- **Value:** `16172031b92b537bca64794096c87b26e96606c6`

#### 7. **LANGSMITH_API_KEY** (선택사항)
- **Name:** `LANGSMITH_API_KEY`
- **Value:** `lsv2_pt_your_langsmith_key` (있으면 입력, 없으면 생략)

#### 8. **LANGSMITH_TRACING** (선택사항)
- **Name:** `LANGSMITH_TRACING`
- **Value:** `true` (또는 `false`)

#### 9. **LANGSMITH_PROJECT** (선택사항)
- **Name:** `LANGSMITH_PROJECT`
- **Value:** `ai-agent` (프로젝트명)

---

## 3️⃣ Secrets 설정 확인

모든 Secrets가 추가되었는지 확인:

**필수:**
```
✅ EC2_HOST
✅ EC2_USERNAME
✅ EC2_SSH_KEY
✅ OPENAI_API_KEY
✅ TAVILY_API_KEY
✅ SERPER_API_KEY
```

**선택 (LangSmith 디버깅용):**
```
⚪ LANGSMITH_API_KEY
⚪ LANGSMITH_TRACING
⚪ LANGSMITH_PROJECT
```

---

## 4️⃣ 첫 배포

### 4-1. 코드를 GitHub에 푸시

```bash
# Git 초기화 (아직 안 했다면)
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 푸시
git push -u origin main
```

### 4-2. GitHub Actions 확인

1. GitHub 저장소로 이동
2. **Actions** 탭 클릭
3. "Deploy to AWS EC2" 워크플로우 확인
4. 진행 상황 모니터링

**예상 소요 시간:** 3-5분

---

## 5️⃣ 자동 배포 테스트

### 5-1. 코드 수정
```bash
# 아무 파일이나 수정
echo "# Test" >> README.md
```

### 5-2. 커밋 및 푸시
```bash
git add .
git commit -m "Test auto deploy"
git push origin main
```

### 5-3. 배포 확인
- GitHub Actions에서 자동으로 배포 시작
- EC2에 자동으로 배포됨
- 브라우저에서 `http://YOUR_EC2_IP:8000` 접속 확인

---

## 6️⃣ 수동 배포 트리거

### GitHub Actions에서 수동 실행

1. GitHub 저장소 → **Actions** 탭
2. "Deploy to AWS EC2" 선택
3. **Run workflow** 클릭
4. **Run workflow** 확인

---

## 7️⃣ 로그 확인

### GitHub Actions 로그
```
GitHub 저장소 → Actions → 워크플로우 선택 → 로그 확인
```

### EC2 로그 (SSH 접속)
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
cd ~/ai-agent
docker-compose logs -f app
```

---

## 8️⃣ 문제 해결

### ❌ "Permission denied (publickey)"
**원인:** SSH 키가 잘못 설정됨

**해결:**
1. EC2_SSH_KEY Secret 확인
2. 키 파일 전체 내용이 복사되었는지 확인
3. 줄바꿈 포함 여부 확인

### ❌ "Could not resolve hostname"
**원인:** EC2_HOST가 잘못 설정됨

**해결:**
1. EC2 Public IP 확인
2. EC2_HOST Secret에 올바른 IP 입력

### ❌ Docker 빌드 실패
**원인:** 메모리 부족 또는 종속성 문제

**해결:**
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
cd ~/ai-agent
docker system prune -a
docker-compose build --no-cache
```

### ❌ 포트 충돌
**원인:** 포트 8000이 이미 사용 중

**해결:**
```bash
sudo lsof -i :8000
sudo kill -9 PID
```

---

## 9️⃣ 배포 흐름 상세

```mermaid
graph TD
    A[코드 푸시] --> B[GitHub Actions 트리거]
    B --> C[코드 체크아웃]
    C --> D[SSH 키 설정]
    D --> E[.env 파일 생성]
    E --> F[코드 동기화]
    F --> G[Docker 이미지 빌드]
    G --> H[컨테이너 재시작]
    H --> I[헬스 체크]
    I --> J{성공?}
    J -->|Yes| K[배포 완료 ✅]
    J -->|No| L[배포 실패 ❌]
```

---

## 🔐 보안 권장사항

### 1. Secrets 관리
- ❌ 절대 코드에 하드코딩 금지
- ✅ GitHub Secrets만 사용
- ✅ 주기적으로 API 키 갱신

### 2. SSH 키 관리
- ✅ EC2 전용 키 페어 생성
- ✅ 키 파일 권한 `chmod 400`
- ❌ 절대 Git에 커밋 금지

### 3. 보안 그룹
- ✅ 필요한 포트만 오픈 (22, 8000)
- ✅ SSH는 특정 IP만 허용 (선택사항)

---

## 📊 배포 모니터링

### 실시간 로그
```bash
# EC2 접속
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 로그 확인
cd ~/ai-agent
docker-compose logs -f app
```

### 컨테이너 상태
```bash
docker-compose ps
docker stats
```

### 리소스 사용량
```bash
free -h
df -h
```

---

## ✅ 체크리스트

### 초기 설정
- [ ] EC2 인스턴스 생성
- [ ] Docker 설치
- [ ] ~/ai-agent 디렉토리 생성
- [ ] GitHub Secrets 6개 모두 설정

### 매 배포
- [ ] 코드 변경
- [ ] Git commit & push
- [ ] GitHub Actions 확인
- [ ] 배포 완료 확인
- [ ] URL 접속 테스트

---

## 🎉 완료!

이제 코드를 푸시하면 자동으로 배포됩니다!

```bash
git add .
git commit -m "Update feature"
git push origin main
```

**배포 URL:** `http://YOUR_EC2_IP:8000`

**GitHub Actions:** https://github.com/YOUR_USERNAME/YOUR_REPO/actions

