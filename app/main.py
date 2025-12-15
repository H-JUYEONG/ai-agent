from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.chat import router as chat_router


app = FastAPI(title="AI Agent Chat")


# Static 파일 마운트
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# Router 등록
app.include_router(chat_router)