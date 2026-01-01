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
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app.mount("/static", NoCacheStaticFiles(directory="app/static"), name="static")

# Router 등록
app.include_router(chat_router)
