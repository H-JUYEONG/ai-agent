# AWS EC2 Docker 배포 가이드 🚀

## 📋 목차
1. [EC2 인스턴스 생성](#1-ec2-인스턴스-생성)
2. [보안 그룹 설정](#2-보안-그룹-설정)
3. [EC2 접속 및 초기 설정](#3-ec2-접속-및-초기-설정)
4. [Docker 설치](#4-docker-설치)
5. [애플리케이션 배포](#5-애플리케이션-배포)
6. [모니터링 및 관리](#6-모니터링-및-관리)

---

## 1. EC2 인스턴스 생성

### 1-1. AWS Console 접속
- https://console.aws.amazon.com/ec2
- **Instances** → **Launch Instance** 클릭

### 1-2. 인스턴스 설정

**이름:**
```
ai-agent-server
```

**AMI (Amazon Machine Image):**
```
Ubuntu Server 22.04 LTS (Free tier eligible)
```

**인스턴스 타입:**
```
t2.medium (권장)
- vCPU: 2
- Memory: 4GB
- 가격: 약 $0.0464/시간

또는

t2.small (최소)
- vCPU: 1
- Memory: 2GB
- 가격: 약 $0.023/시간
```

**키 페어:**
- 새로 생성: `ai-agent-key`
- 파일 저장: `ai-agent-key.pem`
- **⚠️ 중요: 이 파일을 안전하게 보관하세요!**

**스토리지:**
```
30 GB gp3 (권장)
```

---

## 2. 보안 그룹 설정

### 2-1. 인바운드 규칙 추가

| 타입 | 프로토콜 | 포트 | 소스 | 설명 |
|------|----------|------|------|------|
| SSH | TCP | 22 | My IP | SSH 접속 |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 | 웹 애플리케이션 |
| Custom TCP | TCP | 6379 | 172.31.0.0/16 | Redis (내부) |

---

## 3. EC2 접속 및 초기 설정

### 3-1. SSH 접속

**Windows (PowerShell/WSL):**
```bash
# 키 파일 권한 설정
chmod 400 ai-agent-key.pem

# EC2 접속
ssh -i ai-agent-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

**예시:**
```bash
ssh -i ai-agent-key.pem ubuntu@3.35.123.45
```

### 3-2. 시스템 업데이트

```bash
# 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y curl git vim
```

---

## 4. Docker 설치

### 4-1. Docker 설치 스크립트

```bash
# Docker 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자를 docker 그룹에 추가
sudo usermod -aG docker ubuntu

# 로그아웃 후 재로그인 (또는 재시작)
exit
```

재접속:
```bash
ssh -i ai-agent-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

### 4-2. Docker Compose 설치

```bash
# Docker Compose 최신 버전 설치
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 실행 권한 부여
sudo chmod +x /usr/local/bin/docker-compose

# 버전 확인
docker --version
docker-compose --version
```

**예상 출력:**
```
Docker version 24.0.7, build afdd53b
Docker Compose version v2.23.3
```

---

## 5. 애플리케이션 배포

### 5-1. 코드 업로드

**방법 1: Git (권장)**
```bash
# Git 저장소 클론
git clone https://github.com/YOUR_USERNAME/ai-agent.git
cd ai-agent
```

**방법 2: SCP로 직접 업로드**
```bash
# 로컬에서 실행 (Windows PowerShell)
scp -i ai-agent-key.pem -r C:\Users\juyeongzz\Desktop\ai-agent ubuntu@YOUR_EC2_PUBLIC_IP:~/
```

### 5-2. 환경 변수 설정

```bash
# .env 파일 생성
cd ~/ai-agent
nano .env
```

**.env 파일 내용:**
```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Tavily API Key
TAVILY_API_KEY=your_tavily_api_key_here

# Serper API Key
SERPER_API_KEY=16172031b92b537bca64794096c87b26e96606c6

# Redis (Docker Compose가 자동 설정)
REDIS_HOST=redis
REDIS_PORT=6379
```

**저장:** `Ctrl+O` → `Enter` → `Ctrl+X`

### 5-3. 배포 스크립트 실행

```bash
# 실행 권한 부여
chmod +x deploy.sh

# 배포 실행
./deploy.sh
```

**또는 수동 배포:**
```bash
# Docker 이미지 빌드
docker-compose build

# 컨테이너 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

---

## 6. 모니터링 및 관리

### 6-1. 서비스 상태 확인

```bash
# 컨테이너 상태
docker-compose ps

# 로그 확인
docker-compose logs app
docker-compose logs redis

# 실시간 로그
docker-compose logs -f app
```

### 6-2. 애플리케이션 접속

**브라우저에서 접속:**
```
http://YOUR_EC2_PUBLIC_IP:8000
```

**API 테스트:**
```bash
curl http://YOUR_EC2_PUBLIC_IP:8000/
```

### 6-3. 유용한 명령어

```bash
# 컨테이너 재시작
docker-compose restart

# 컨테이너 중지
docker-compose stop

# 컨테이너 시작
docker-compose start

# 컨테이너 중지 및 제거
docker-compose down

# 리소스 사용량 확인
docker stats

# 디스크 사용량 확인
df -h

# 메모리 사용량 확인
free -h
```

### 6-4. 업데이트 배포

```bash
# Git으로 최신 코드 받기
git pull origin main

# 배포 스크립트 실행
./deploy.sh
```

---

## 7. 문제 해결

### 7-1. 포트 8000이 사용 중인 경우
```bash
# 포트 사용 프로세스 확인
sudo lsof -i :8000

# 프로세스 종료
sudo kill -9 PID
```

### 7-2. Docker 컨테이너가 시작되지 않는 경우
```bash
# 로그 상세 확인
docker-compose logs app

# 컨테이너 재빌드
docker-compose build --no-cache
docker-compose up -d
```

### 7-3. Redis 연결 오류
```bash
# Redis 컨테이너 확인
docker-compose ps redis

# Redis 재시작
docker-compose restart redis
```

### 7-4. 메모리 부족
```bash
# 메모리 확인
free -h

# 불필요한 Docker 이미지 제거
docker system prune -a
```

---

## 8. 보안 권장사항

### 8-1. 방화벽 설정
```bash
# UFW 활성화
sudo ufw enable

# SSH 허용
sudo ufw allow 22/tcp

# 애플리케이션 포트 허용
sudo ufw allow 8000/tcp

# 상태 확인
sudo ufw status
```

### 8-2. 자동 업데이트 설정
```bash
# 자동 보안 업데이트
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 8-3. SSL/HTTPS 설정 (선택사항)
```bash
# Nginx + Let's Encrypt 사용 권장
# 별도 가이드 필요 시 요청
```

---

## 9. 비용 최적화

### 9-1. EC2 인스턴스 중지/시작
```bash
# AWS Console에서:
# Instances → 선택 → Instance state → Stop
# (사용하지 않을 때 중지하면 컴퓨팅 비용 절약)
```

### 9-2. 예상 비용
- **t2.small**: ~$17/월
- **t2.medium**: ~$34/월
- **데이터 전송**: 최초 100GB 무료

---

## 10. 도메인 연결 (선택사항)

### 10-1. Route 53 설정
1. AWS Route 53에서 도메인 구매
2. A 레코드 생성: EC2 Public IP 연결
3. `http://yourdomain.com:8000` 접속

### 10-2. Nginx 리버스 프록시 (포트 80/443)
```bash
# Nginx 설치
sudo apt install nginx

# 설정 파일 생성
sudo nano /etc/nginx/sites-available/ai-agent
```

---

## ✅ 배포 완료 체크리스트

- [ ] EC2 인스턴스 생성
- [ ] 보안 그룹 설정 (포트 22, 8000)
- [ ] SSH 접속 성공
- [ ] Docker 설치 완료
- [ ] Docker Compose 설치 완료
- [ ] 코드 업로드 완료
- [ ] .env 파일 설정 완료
- [ ] 배포 스크립트 실행 성공
- [ ] 브라우저에서 접속 확인
- [ ] API 테스트 성공

---

## 🆘 지원

문제가 발생하면 다음을 확인하세요:
1. `docker-compose logs app` - 애플리케이션 로그
2. `docker-compose logs redis` - Redis 로그
3. AWS EC2 콘솔 - 인스턴스 상태
4. 보안 그룹 - 인바운드 규칙

**완료!** 🎉


