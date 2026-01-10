#!/bin/bash

# AI Agent AWS EC2 배포 스크립트
# 사용법: ./deploy.sh

set -e

echo "🚀 AI Agent 배포 시작..."

# 1. 최신 코드 pull (Git 사용 시)
# echo "📥 코드 업데이트..."
# git pull origin main

# 2. Redis 컨테이너 확인 및 시작 (없을 경우에만 생성)
echo "🔍 Redis 컨테이너 확인..."
if ! docker compose ps redis | grep -q "Up"; then
    echo "📦 Redis 컨테이너 시작..."
    docker compose up -d redis
    echo "⏳ Redis 초기화 대기..."
    sleep 3
else
    echo "✅ Redis 이미 실행 중"
fi

# 2-1. Qdrant 컨테이너 확인 및 시작 (없을 경우에만 생성)
echo "🔍 Qdrant 컨테이너 확인..."
if ! docker compose ps qdrant | grep -q "Up"; then
    echo "📦 Qdrant 컨테이너 시작..."
    docker compose up -d qdrant
    echo "⏳ Qdrant 초기화 대기..."
    sleep 5
else
    echo "✅ Qdrant 이미 실행 중"
fi

# 3. Redis 캐시 초기화 (키만 삭제)
echo "🗑️ Redis 캐시 초기화..."
docker compose exec -T redis redis-cli FLUSHALL

# 4. 애플리케이션 컨테이너만 중지 및 제거
echo "🛑 애플리케이션 컨테이너 중지 및 제거..."
docker compose stop app
docker compose rm -f app

# 5. 사용하지 않는 이미지 정리
echo "🧹 불필요한 Docker 이미지 정리..."
docker image prune -a -f

# 6. 애플리케이션 이미지 빌드 (빌드 ID로 캐시 무효화)
echo "🔨 애플리케이션 이미지 빌드..."
export BUILD_ID=$(date +%s)
export BUILD_TIME=$(date -u +%Y%m%d-%H%M%S)
echo "📅 Build ID: $BUILD_ID, Build Time: $BUILD_TIME"
docker compose build --no-cache app

# 7. 애플리케이션 컨테이너 시작
echo "▶️ 애플리케이션 컨테이너 시작..."
docker compose up -d app

# 6. 로그 확인
echo "📋 로그 확인 (최근 로그)..."
sleep 10
docker compose logs app --tail=100

# 7. 헬스 체크 (최대 5분 대기)
echo "🏥 헬스 체크 - App이 준비될 때까지 대기 중..."
echo "Waiting for app to be ready..."

for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1 || curl -sf http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ App is ready!"
        echo "✅ 배포 성공! 서비스가 정상 작동 중입니다."
        echo "🌐 URL: http://$(curl -s ifconfig.me):8000"
        exit 0
    fi
    echo "⏳ Not ready yet... retry $i/30"
    sleep 10
done

echo "❌ Health check failed"
echo "❌ 배포 실패! 로그를 확인하세요."
echo "📋 App 컨테이너 로그:"
docker compose logs app --tail=50
echo "📋 전체 컨테이너 상태:"
docker compose ps
exit 1

echo "📊 실행 중인 컨테이너:"
docker compose ps

echo "✨ 배포 완료!"
