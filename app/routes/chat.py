from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import os
import time

from app.agent.graph import deep_researcher
from app.tools.cache import research_cache


router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


class ChatRequest(BaseModel):
    message: str
    domain: str = "코딩"  # 항상 코딩으로 고정
    history: list = []  # 이전 대화 이력


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """팀 상황 기반 코딩 AI 도입 의사결정 에이전트 API"""
    
    try:
        # 도메인은 항상 코딩으로 고정
        domain = "코딩"
        
        # 시작 시간 기록
        start_time = time.time()
        
        # 1. 캐시 확인
        cached_result = research_cache.get(req.message, domain)
        if cached_result:
            elapsed_time = time.time() - start_time
            print(f"✅ 캐시에서 응답 반환 (소요 시간: {elapsed_time:.2f}초)")
            return {"reply": cached_result["reply"]}
        
        # 2. LangGraph 실행
        print(f"🔍 Deep Research 시작: {req.message} (도메인: {domain})")
        
        # 환경 변수 확인
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key or openai_key.startswith("sk-proj-xxx"):
            return {
                "reply": "⚠️ OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
            }
        
        # 대화 이력 구성
        messages_to_send = []
        
        # 이전 대화 이력 추가
        for msg in req.history:
            if msg.get("role") == "user":
                messages_to_send.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages_to_send.append(AIMessage(content=msg.get("content", "")))
        
        # 현재 사용자 메시지 추가
        messages_to_send.append(HumanMessage(content=req.message))
        
        print(f"🔍 [DEBUG] chat.py - 전송할 Messages 개수: {len(messages_to_send)}개")
        
        # LangGraph 실행
        result = await deep_researcher.ainvoke(
            {
                "messages": messages_to_send,
                "domain": domain
            },
            config={
                "configurable": {
                    "domain": domain
                }
            }
        )
        
        # 최종 리포트 추출
        messages = result.get("messages", [])
        
        # AI 메시지만 추출 (마지막 N개)
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        
        # 마지막 2개의 AI 메시지 확인 (인사말 + 리포트 분리)
        if len(ai_messages) >= 2:
            # 마지막 2개 메시지 반환 (리스트로)
            reply_messages = [ai_messages[-2].content, ai_messages[-1].content]
            print(f"✅ [DEBUG] 2개 메시지 감지: {len(reply_messages)}개")
        elif len(ai_messages) == 1:
            # 1개만 있으면 그대로 반환
            reply_messages = [ai_messages[-1].content]
            print(f"✅ [DEBUG] 1개 메시지 감지")
        else:
            # 메시지 없으면 final_report 사용
            final_report = result.get("final_report", "")
            reply_messages = [final_report] if final_report else ["응답을 생성하지 못했습니다."]
        
        # 3. 캐시 저장 (마지막 메시지만)
        cache_data = {"reply": reply_messages[-1] if reply_messages else ""}
        research_cache.set(req.message, cache_data, domain)
        
        # 종료 시간 계산
        elapsed_time = time.time() - start_time
        print(f"✅ Deep Research 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        # 여러 메시지면 배열로, 하나면 문자열로 반환
        if len(reply_messages) > 1:
            return {"reply": reply_messages}
        else:
            return {"reply": reply_messages[0]}
    
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