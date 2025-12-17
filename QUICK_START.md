# 빠른 시작 가이드 ⚡

## 🚀 5분 안에 배포하기

### 1️⃣ EC2 인스턴스 준비 (AWS Console)

```
- AMI: Ubuntu 22.04 LTS
- 인스턴스 타입: t2.medium
- 보안 그룹: 포트 22, 8000 오픈
- 키 페어: 다운로드 및 보관
```

### 2️⃣ SSH 접속

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

### 3️⃣ Docker 설치 (원라인)

```bash
curl -fsSL https://get.docker.com | sudo sh && \
sudo usermod -aG docker ubuntu && \
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
sudo chmod +x /usr/local/bin/docker-compose
```

**재접속:**
```bash
exit
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

### 4️⃣ 코드 업로드

**방법 A: Git**
```bash
git clone YOUR_REPO_URL
cd ai-agent
```

**방법 B: SCP (로컬에서 실행)**
```bash
scp -i your-key.pem -r C:\Users\juyeongzz\Desktop\ai-agent ubuntu@YOUR_EC2_IP:~/
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
cd ai-agent
```

### 5️⃣ 환경 변수 설정

```bash
cat > .env << EOF
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
SERPER_API_KEY=16172031b92b537bca64794096c87b26e96606c6
REDIS_HOST=redis
REDIS_PORT=6379
EOF
```

### 6️⃣ 배포!

```bash
chmod +x deploy.sh
./deploy.sh
```

### 7️⃣ 접속 확인

```
http://YOUR_EC2_IP:8000
```

---

## 🎯 핵심 명령어

```bash
# 로그 확인
docker-compose logs -f app

# 재시작
docker-compose restart

# 중지
docker-compose down

# 시작
docker-compose up -d

# 상태 확인
docker-compose ps
```

---

## 🆘 문제 해결

**컨테이너가 안 뜰 때:**
```bash
docker-compose logs app
docker-compose build --no-cache
docker-compose up -d
```

**포트 충돌:**
```bash
sudo lsof -i :8000
sudo kill -9 PID
```

**메모리 부족:**
```bash
docker system prune -a
```

---

## ✅ 완료!

브라우저에서 `http://YOUR_EC2_IP:8000` 접속하면 AI Agent가 작동합니다! 🎉


