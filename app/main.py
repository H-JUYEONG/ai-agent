from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.chat import router as chat_router
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

app = FastAPI(title="AI Agent Chat")


# Health Check (Docker / CI)
@app.get("/health")
def health():
    return {"status": "ok"}

# Static 파일 (캐시 비활성화)
from fastapi.responses import FileResponse
from starlette.staticfiles import StaticFiles as StarletteStaticFiles

class NoCacheStaticFiles(StarletteStaticFiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        # 캐시 완전 비활성화
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        # ETag와 Last-Modified 제거 (브라우저가 304 Not Modified를 반환하지 않도록)
        if "ETag" in response.headers:
            del response.headers["ETag"]
        if "Last-Modified" in response.headers:
            del response.headers["Last-Modified"]
        return response

app.mount("/static", NoCacheStaticFiles(directory="app/static"), name="static")

# Router 등록
app.include_router(chat_router)
