# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 빌드 인자 (캐시 무효화용)
ARG BUILD_ID
ARG BUILD_TIME

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip setuptools wheel

# torch CPU 버전 먼저 설치 (CUDA 버전은 너무 큼, 디스크 공간 절약)
RUN pip install --no-cache-dir "torch>=2.0.0,<3.0.0" --index-url https://download.pytorch.org/whl/cpu

# Python 패키지 설치 (캐시 활용, 타임아웃 증가)
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --default-timeout=100 -r /tmp/requirements.txt

# 모델 preload (컨테이너 시작 시 모델 로딩 시간 단축)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || echo "Model preload failed, will load at runtime"

# 빌드 중 사용한 임시 파일 정리 (디스크 공간 확보)
RUN apt-get clean && \
    rm -rf /tmp/* /var/tmp/* && \
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 애플리케이션 코드 복사 (레이어 캐싱 최적화)
# Python 코드 (자주 변경)
COPY app/agent /app/app/agent
COPY app/routes /app/app/routes
COPY app/tools /app/app/tools
COPY app/main.py /app/app/
COPY app/__init__.py /app/app/

# 프론트엔드 (static & templates) - 별도 레이어로 분리
COPY app/static /app/app/static
COPY app/templates /app/app/templates

# 기타 파일
COPY check_storage.py /app/

# 빌드 정보 저장 (캐시 무효화 보장)
RUN echo "BUILD_ID=${BUILD_ID:-$(date +%s)}" > /app/build-info.txt && \
    echo "BUILD_TIME=${BUILD_TIME:-$(date -u +%Y%m%d-%H%M%S)}" >> /app/build-info.txt

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



