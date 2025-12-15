from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


router = APIRouter()
templates = Jinja2Templates(directory="app/templates")




class ChatRequest(BaseModel):
    message: str




@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )




@router.post("/api/chat")
async def chat(req: ChatRequest):
    # TODO: LangGraph + Deep Research 연결 위치
    return {
        "reply": f"(임시 응답) 질문을 잘 받았습니다. {req.message}"
    }