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
        
        # 캐시 확인은 clarify_with_user에서 처리 (인사 멘트 생성 포함)
        # LangGraph 실행
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
        
        # AI 메시지만 추출 (중복 제거)
        ai_messages = []
        seen_contents = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                # 중복 제거: 동일한 내용이 이미 있으면 스킵
                if content not in seen_contents:
                    ai_messages.append(msg)
                    seen_contents.add(content)
        
        # 마지막 메시지 확인 (인사말 + 리포트 분리)
        reply_messages = []
        if len(ai_messages) >= 2:
            # 마지막 2개 메시지 확인
            last_msg = ai_messages[-1].content
            second_last_msg = ai_messages[-2].content
            
            # 인사 멘트는 보통 짧고(200자 미만), 리포트는 길다(200자 이상)
            # 마지막 2개 메시지가 인사 멘트 + 리포트 조합인지 확인
            if len(second_last_msg) < 200 and len(last_msg) >= 200:
                # 인사 멘트(짧음) + 리포트(길음) 조합
                reply_messages = [second_last_msg, last_msg]
                print(f"✅ [DEBUG] 인사말 + 리포트 분리: 2개 (인사말: {len(second_last_msg)}자, 리포트: {len(last_msg)}자)")
            elif len(second_last_msg) >= 200 and len(last_msg) < 200:
                # 리포트(길음) + 인사 멘트(짧음) 순서 (순서가 바뀐 경우)
                reply_messages = [last_msg, second_last_msg]
                print(f"✅ [DEBUG] 리포트 + 인사말 분리: 2개 (순서 변경, 인사말: {len(last_msg)}자, 리포트: {len(second_last_msg)}자)")
            else:
                # 둘 다 길거나 둘 다 짧으면 마지막 2개 모두 반환 (안전하게)
                reply_messages = [second_last_msg, last_msg]
                print(f"✅ [DEBUG] 마지막 2개 메시지 모두 반환: 2개")
        elif len(ai_messages) == 1:
            reply_messages = [ai_messages[-1].content]
            print(f"✅ [DEBUG] 1개 메시지 감지")
        else:
            # 메시지 없으면 final_report 사용
            final_report = result.get("final_report", "")
            reply_messages = [final_report] if final_report else ["응답을 생성하지 못했습니다."]
        
        # 3. 캐시 저장 (주제에 맞는 질문만 저장)
        last_reply = reply_messages[-1] if reply_messages else ""
        
        # 주제에서 벗어난 거부 메시지는 캐시하지 않음
        is_off_topic_rejection = (
            "죄송합니다" in last_reply and 
            "코딩 AI 도구 추천을 전문으로" in last_reply
        )
        
        if not is_off_topic_rejection:
            cache_data = {"reply": last_reply}
            research_cache.set(req.message, cache_data, domain)
            cache_type = "Redis" if research_cache.available else "메모리"
            print(f"💾 {cache_type} 캐시 저장: {req.message[:50]}...")
        else:
            print(f"⚠️ 주제 벗어난 질문 - 캐시 저장 생략")
        
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