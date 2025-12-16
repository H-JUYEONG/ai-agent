from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import os
import time

from app.agent.graph import deep_researcher
from app.tools.cache import research_cache


router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


class ChatRequest(BaseModel):
    message: str
    domain: str = "LLM"  # LLM, 코딩, 디자인


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """AI 서비스 비교 분석 챗봇 API"""
    
    try:
        # 시작 시간 기록
        start_time = time.time()
        
        # 1. 캐시 확인
        cached_result = research_cache.get(req.message, req.domain)
        if cached_result:
            elapsed_time = time.time() - start_time
            print(f"✅ 캐시에서 응답 반환 (소요 시간: {elapsed_time:.2f}초)")
            return {"reply": cached_result["reply"]}
        
        # 2. LangGraph 실행
        print(f"🔍 Deep Research 시작: {req.message} (도메인: {req.domain})")
        
        # 환경 변수 확인
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key or openai_key.startswith("sk-proj-xxx"):
            return {
                "reply": "⚠️ OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
            }
        
        # LangGraph 실행
        result = await deep_researcher.ainvoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "domain": req.domain
            },
            config={
                "configurable": {
                    "domain": req.domain
                }
            }
        )
        
        # 최종 리포트 추출
        final_report = result.get("final_report", "")
        
        if not final_report:
            # messages에서 마지막 AI 메시지 추출
            messages = result.get("messages", [])
            if messages:
                final_report = messages[-1].content
        
        # 3. 캐시 저장
        cache_data = {"reply": final_report}
        research_cache.set(req.message, cache_data, req.domain)
        
        # 종료 시간 계산
        elapsed_time = time.time() - start_time
        print(f"✅ Deep Research 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return {"reply": final_report}
    
    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}"
        print(error_msg)
        return {
            "reply": f"""
죄송합니다. 처리 중 오류가 발생했습니다.

**오류 내용:**
{str(e)}

**해결 방법:**
1. .env 파일에 API 키가 올바르게 설정되었는지 확인
2. 인터넷 연결 확인
3. 질문을 더 간단하게 작성해보기

다시 시도해주세요.
            """
        }