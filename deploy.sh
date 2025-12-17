#!/bin/bash

# AI Agent AWS EC2 배포 스크립트
# 사용법: ./deploy.sh

set -e

echo "🚀 AI Agent 배포 시작..."

# 1. 최신 코드 pull (Git 사용 시)
# echo "📥 코드 업데이트..."
# git pull origin main

# 2. 기존 컨테이너 중지 및 제거
echo "🛑 기존 컨테이너 중지 및 제거..."
docker compose down

# 3. 사용하지 않는 이미지 및 컨테이너 정리
echo "🧹 불필요한 Docker 리소스 정리..."
docker system prune -f
docker image prune -a -f

# 4. 새 이미지 빌드
echo "🔨 Docker 이미지 빌드..."
docker compose build --no-cache

# 5. 컨테이너 시작
echo "▶️ 컨테이너 시작..."
docker compose up -d

# 6. 로그 확인
echo "📋 로그 확인 (최근 로그)..."
sleep 5
docker compose logs --tail=50

# 7. 헬스 체크
echo "🏥 헬스 체크..."
sleep 5

if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ 배포 성공! 서비스가 정상 작동 중입니다."
    echo "🌐 URL: http://$(curl -s ifconfig.me):8000"
else
    echo "❌ 배포 실패! 로그를 확인하세요."
    docker compose logs
    exit 1
fi

echo "📊 실행 중인 컨테이너:"
docker compose ps

echo "✨ 배포 완료!"
