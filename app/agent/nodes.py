"""LangGraph nodes for AI Service Advisor"""

import asyncio
import re
from datetime import datetime
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from app.agent.configuration import Configuration
from app.agent.state import (
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
    ResearcherState,
    SupervisorState,
)
from app.agent.models import ToolFact, UserContext, PricingPlan, SecurityPolicy, WorkflowType
from app.agent.decision import DecisionEngine
from app.agent.fact_extractor import extract_tool_facts
from app.agent.prompts import (
    DOMAIN_GUIDES,
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    final_report_generation_prompt,
    get_today_str,
    get_current_year,
    get_current_month_year,
)
from app.agent.utils import (
    think_tool,
    get_api_key_for_model,
    get_notes_from_tool_calls,
)
from app.tools.search import searcher
from app.tools.vector_store import vector_store
from app.tools.query_normalizer import query_normalizer
from app.tools.cache import research_cache

# 설정 가능한 모델
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """사용자 질문 명확화 및 주제 검증 + 쿼리 정규화 + 캐시 조회"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    configurable = Configuration.from_runnable_config(config)
    messages = state["messages"]
    domain = state.get("domain", "AI 서비스")
    
    # 질문 순서 파악: HumanMessage 개수로 판단 (더 정확하게)
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)  # 1번째, 2번째, 3번째 질문...
    is_followup = question_number > 1  # 2번째 질문부터 Follow-up
    
    # 디버깅
    print(f"🔍 [DEBUG] clarify - Messages: {len(messages)}개, HumanMessage: {len(human_messages)}개, 질문 순서: {question_number}번째, Follow-up: {is_followup}")
    
    # ========== 🆕 1단계: 쿼리 정규화 (캐시 키 생성) ==========
    last_user_message = messages[-1].content if messages else ""
    
    model_config = {
        "model": configurable.research_model,
        "max_tokens": 200,  # 정규화는 짧게
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    normalized = await query_normalizer.normalize(last_user_message, config=model_config)
    cache_key = normalized["cache_key"]
    
    # ========== 🆕 2단계: Redis 최종 답변 캐시 조회 ==========
    print(f"🔍 [캐시 조회] 원본 질문: '{last_user_message[:50]}...'")
    print(f"🔍 [캐시 조회] 정규화: '{normalized['normalized_text']}' → 캐시키: {cache_key[:16]}...")
    
    cached_answer = research_cache.get(cache_key, domain=domain, prefix="final")
    if cached_answer:
        print(f"✅ [캐시 HIT] 최종 답변 반환 (캐시키: {cache_key[:16]}...)")
        
        # Follow-up인 경우 이전 추천 도구 확인
        # 단, 같은 의미의 질문(같은 캐시 키)이면 이전 추천 도구 확인 건너뛰고 캐시 사용
        if is_followup:
            # 이전 메시지에서 추천된 도구 추출 (모든 AI 메시지에서)
            previous_tools_in_messages = []
            all_tools = []
            for msg in reversed(messages[:-1]):  # 마지막 사용자 메시지 제외
                if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                    content = str(msg.content)
                    # 다양한 패턴으로 도구명 추출
                    # 패턴 1: 📊 [도구명]
                    tools_found = re.findall(r'📊\s+([^\n]+)', content)
                    if tools_found:
                        all_tools.extend([t.strip() for t in tools_found])
                    # 패턴 2: ## 📊 [도구명]
                    tools_found2 = re.findall(r'##\s+📊\s+([^\n]+)', content)
                    if tools_found2:
                        all_tools.extend([t.strip() for t in tools_found2])
                    # 패턴 3: **1순위: [도구명]**, **2순위: [도구명]**
                    tools_found3 = re.findall(r'\*\*[0-9]+순위:\s*([^\*]+)\*\*', content)
                    if tools_found3:
                        all_tools.extend([t.strip() for t in tools_found3])
                    # 패턴 4: **최종 추천: [도구명]**
                    tools_found4 = re.findall(r'\*\*최종 추천:\s*([^\*]+)\*\*', content)
                    if tools_found4:
                        all_tools.extend([t.strip() for t in tools_found4])
            
            # 중복 제거
            seen = set()
            for tool in all_tools:
                # 도구명 정제 (불필요한 문자 제거)
                tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
                if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                    seen.add(tool_clean)
                    previous_tools_in_messages.append(tool_clean)
            
            # 이전 추천 도구가 있으면 캐시 검증, 없으면 같은 의미의 질문이므로 캐시 그대로 사용
            if previous_tools_in_messages:
                # 캐시된 답변에서 도구 추출 (다양한 패턴)
                cached_tools = []
                cached_content = cached_answer["content"]
                # 패턴 1: 📊 [도구명]
                tools_found = re.findall(r'📊\s+([^\n]+)', cached_content)
                cached_tools.extend([t.strip() for t in tools_found])
                # 패턴 2: ## 📊 [도구명]
                tools_found2 = re.findall(r'##\s+📊\s+([^\n]+)', cached_content)
                cached_tools.extend([t.strip() for t in tools_found2])
                # 패턴 3: **1순위: [도구명]**
                tools_found3 = re.findall(r'\*\*[0-9]+순위:\s*([^\*]+)\*\*', cached_content)
                cached_tools.extend([t.strip() for t in tools_found3])
                
                # 이전 추천 도구와 캐시된 답변의 도구가 다르면 캐시 무시
                if cached_tools:
                    # 도구명 정제
                    previous_tools_clean = [re.sub(r'[\(\)\[\]월\s\$0-9]+', '', t).strip() for t in previous_tools_in_messages]
                    cached_tools_clean = [re.sub(r'[\(\)\[\]월\s\$0-9]+', '', t).strip() for t in cached_tools]
                    
                    previous_tools_set = set([t for t in previous_tools_clean if len(t) > 2])
                    cached_tools_set = set([t for t in cached_tools_clean if len(t) > 2])
                    
                    # 이전 추천 도구가 캐시에 없거나, 캐시에 이전에 추천하지 않은 새 도구가 있으면 무시
                    if not previous_tools_set.issubset(cached_tools_set) or len(cached_tools_set - previous_tools_set) > 0:
                        print(f"⚠️ [캐시 무시] 이전 추천 도구({previous_tools_in_messages})와 캐시 도구({cached_tools})가 다름. 캐시 무시하고 새로 생성")
                        cached_answer = None  # 캐시 무시
            else:
                # 이전 추천 도구가 없으면 같은 의미의 질문이므로 캐시 그대로 사용
                print(f"✅ [캐시 사용] 이전 추천 도구 없음 - 같은 의미의 질문으로 판단, 캐시 사용")
        
        if cached_answer:
            # 캐시된 답변 처리
            cached_content = cached_answer["content"]
            
            print(f"🔍 [캐시 처리] 캐시된 답변 길이: {len(cached_content)}자, is_followup: {is_followup}")
            print(f"🔍 [캐시 처리] 캐시된 답변 시작 100자: {cached_content[:100]}")
            
            # 리포트 본문 추출 (캐시에는 리포트 본문만 저장되어 있음)
            report_body = cached_content.strip()
            
            # 🚨 [GREETING] 태그가 있으면 제거하고 리포트 본문만 추출
            # 인사 멘트는 캐시에서 가져오지 않고 항상 새로 생성
            if "[GREETING]" in cached_content and "[/GREETING]" in cached_content:
                match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', cached_content, re.DOTALL)
                if match:
                    # 인사말 태그 제거하고 리포트 본문만 추출
                    report_body = cached_content.replace(match.group(0), "").strip()
                    print(f"✅ [캐시] [GREETING] 태그 제거 후 리포트 본문 추출: {len(report_body)}자")
            
            # 리포트 본문이 비어있거나 너무 짧으면 원본 사용
            if not report_body or len(report_body) < 50:
                print(f"⚠️ [캐시 처리] 리포트 본문이 비어있음 - 원본 캐시 내용 사용")
                report_body = cached_content.strip()
            
            # 🚨 캐시 검증: 리포트 본문이 유효한지 확인
            # 리포트가 너무 짧거나(200자 미만) 비어있으면 캐시 무시
            if len(report_body) < 200:
                print(f"⚠️ [캐시 무시] 리포트 본문이 너무 짧음 ({len(report_body)}자). 캐시 무시하고 새로 생성")
                # pass - 캐시를 사용하지 않고 아래 연구 프로세스로 진행
            else:
                # 🚨 인사 멘트는 항상 새로 생성 (캐시에서 가져오지 않음)
                # final_report_generation과 동일한 방식으로 인사 멘트 생성 (동일한 모델, 동일한 프롬프트 스타일)
                print(f"✅ [캐시 처리] 리포트 본문은 캐시에서 가져옴 ({len(report_body)}자), 인사 멘트는 final_report_generation과 동일한 방식으로 생성")
                
                # final_report_generation과 동일한 모델 및 설정 사용
                greeting_model_config = {
                    "model": configurable.final_report_model,
                    "max_tokens": configurable.final_report_model_max_tokens,
                    "api_key": get_api_key_for_model(configurable.final_report_model, config),
                }
                
                # final_report_generation 프롬프트의 인사 멘트 생성 부분과 동일한 스타일
                # 사용자 메시지 전체 컨텍스트 제공 (final_report_generation과 동일)
                messages_context = get_buffer_string(messages) if messages else last_user_message
                
                greeting_prompt = f"""당신은 코딩 AI 도구 추천 전문가입니다. 사용자 질문에 맞는 자연스럽고 상세한 인사 멘트를 생성하세요.

사용자 메시지:
{messages_context}

**원칙:**
- 사용자의 현재 질문 내용과 의도를 정확히 파악하여 그에 맞는 자연스러운 멘트를 생성
- 질문의 핵심 키워드(팀 규모, 목적, 요구사항, 도메인 등)를 반영
- 질문에 언급된 구체적인 내용(팀 규모, 목적, 요구사항 등)을 반드시 포함
- 자연스럽고 친절한 톤 유지
- 적절한 길이 (40-100자 정도, 너무 짧지 않게)

**좋은 예시:**
- 질문: "저희는 백엔드·프론트엔드 포함해서 8명 규모의 개발팀인데, 코드 작성과 리뷰에 AI를 도입해서 생산성을 높이고 싶습니다. 어떤 도구가 좋을까요?"
  인사 멘트: "네! 백엔드와 프론트엔드를 포함한 8명 규모의 개발팀에 적합한 AI 도구들을 분석해드리겠습니다. 팀의 코드 작성 및 리뷰 효율성 향상에 도움이 되는 도구를 비교해드리겠습니다."

- 질문: "코드 작성과 리뷰를 위한 AI 도구 추천해줘"
  인사 멘트: "네! 코드 작성과 리뷰를 위한 최적의 AI 도구를 추천해드리겠습니다."

**나쁜 예시 (너무 짧거나 맥락 없음):**
- "안녕하세요." (너무 짧음)
- "AI 도구로 생산성을 높여드리겠습니다." (너무 짧고 구체적이지 않음)
- "네! 조사해드리겠습니다." (너무 일반적)

인사 멘트만 출력하세요 ([GREETING] 태그 없이, 다른 설명 없이):"""
                
                try:
                    greeting_model = configurable_model.with_config(greeting_model_config)
                    greeting_response = await greeting_model.ainvoke([HumanMessage(content=greeting_prompt)])
                    greeting = str(greeting_response.content).strip()
                    
                    # 불필요한 따옴표나 태그 제거
                    greeting = greeting.strip('"\'`').strip()
                    
                    # "안녕하세요"로만 시작하는 너무 짧은 응답 감지
                    if greeting.startswith("안녕하세요") and len(greeting) < 15:
                        print(f"⚠️ [캐시 처리] LLM 응답이 너무 짧음: '{greeting}', 재시도")
                        greeting = ""  # 재시도하도록 빈 문자열로 설정
                    
                    # 응답이 너무 길면 적절히 자르기 (100자 이내로)
                    if greeting and len(greeting) > 100:
                        # 문장 단위로 자르기 (마침표나 느낌표 기준)
                        sentences = re.split(r'[.!?。]', greeting)
                        if len(sentences) > 1 and sentences[0]:
                            # 첫 번째 문장만 사용하고 마침표 추가
                            greeting = sentences[0].strip() + '.'
                        else:
                            # 문장 구분이 없으면 100자로 자르기
                            greeting = greeting[:100].strip()
                    
                    # 빈 응답이거나 너무 짧으면 재시도 (최소 30자 이상)
                    if not greeting or len(greeting) < 30:
                        print(f"⚠️ [캐시 처리] LLM 응답이 너무 짧음 ({len(greeting) if greeting else 0}자), 재시도")
                        # 더 상세한 프롬프트로 재시도 (final_report_generation 스타일)
                        retry_prompt = f"""당신은 코딩 AI 도구 추천 전문가입니다.

사용자 메시지:
{messages_context}

위 질문에 맞는 자연스럽고 상세한 인사 멘트를 생성하세요. 질문의 핵심 내용(팀 규모, 목적, 요구사항 등)을 구체적으로 반영한 40-100자 정도의 상세한 인사 멘트를 작성해주세요.

예시: "네! 백엔드와 프론트엔드를 포함한 8명 규모의 개발팀에 적합한 AI 도구들을 분석해드리겠습니다. 팀의 코드 작성 및 리뷰 효율성 향상에 도움이 되는 도구를 비교해드리겠습니다."

인사 멘트만 출력하세요:"""
                        retry_response = await greeting_model.ainvoke([HumanMessage(content=retry_prompt)])
                        greeting = str(retry_response.content).strip().strip('"\'`').strip()
                        
                        # 재시도 후에도 너무 짧으면 질문 기반으로 동적 생성
                        if not greeting or len(greeting) < 30:
                            # 질문의 핵심 키워드를 추출해서 동적으로 생성
                            keywords = []
                            if "팀" in last_user_message or "규모" in last_user_message:
                                keywords.append("팀")
                            if "코드" in last_user_message or "리뷰" in last_user_message:
                                keywords.append("코드 작성 및 리뷰")
                            if "도구" in last_user_message or "추천" in last_user_message:
                                keywords.append("도구 추천")
                            
                            if keywords:
                                greeting = f"네! {'와 '.join(keywords[:2])}에 적합한 AI 도구를 분석해드리겠습니다."
                            else:
                                greeting = f"네! {last_user_message[:30]}에 대해 조사해드리겠습니다."
                    
                    print(f"✅ [캐시 처리] LLM으로 인사 멘트 생성 완료: '{greeting}' (길이: {len(greeting)}자)")
                except Exception as e:
                    print(f"⚠️ [캐시 처리] LLM 인사 멘트 생성 실패: {e}, 질문 기반 동적 생성")
                    # LLM 실패 시 질문 내용을 기반으로 동적으로 생성 (하드코딩 최소화)
                    question_preview = last_user_message[:50] if len(last_user_message) > 50 else last_user_message
                    greeting = f"네! {question_preview}에 대해 조사해드리겠습니다."
                    print(f"✅ [캐시 처리] 동적 생성 인사 멘트: '{greeting}'")
                print(f"✅ [캐시 처리] 리포트 본문 길이: {len(report_body)}자, 시작 100자: {report_body[:100]}")
                
                return Command(
                    goto="__end__",
                    update={"messages": [
                        AIMessage(content=greeting),
                        AIMessage(content=report_body)
                    ]}
                )
    
    print(f"⚠️ [캐시 MISS] 정규화된 쿼리: '{normalized['normalized_text']}' (키워드: {normalized['keywords']})")
    
    # ========== 🆕 3단계: 벡터 DB로 유사 질문 검색 ==========
    # 캐시 미스 시 유사한 질문이 있는지 벡터 DB에서 검색
    similar_query = vector_store.search_similar_query(
        query=last_user_message,
        domain=domain,
        limit=1,
        score_threshold=0.85  # 높은 유사도만 (85% 이상)
    )
    
    if similar_query and similar_query.get("cache_key"):
        similar_cache_key = similar_query["cache_key"]
        print(f"🔍 [유사 질문 발견] 유사도: {similar_query['score']:.3f}, 기존 질문: '{similar_query['query'][:50]}...'")
        print(f"🔍 [유사 질문] 캐시 키 재사용: {similar_cache_key[:16]}...")
        
        # 유사 질문의 캐시 키로 Redis에서 답변 가져오기
        cached_answer = research_cache.get(similar_cache_key, domain=domain, prefix="final")
        if cached_answer:
            print(f"✅ [유사 질문 캐시 HIT] 최종 답변 반환 (유사 질문의 캐시 키: {similar_cache_key[:16]}...)")
            
            # 리포트 본문 추출 및 인사 멘트 생성 (기존 로직과 동일)
            cached_content = cached_answer["content"]
            report_body = cached_content.strip()
            
            # [GREETING] 태그 제거
            if "[GREETING]" in cached_content and "[/GREETING]" in cached_content:
                match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', cached_content, re.DOTALL)
                if match:
                    report_body = cached_content.replace(match.group(0), "").strip()
            
            if not report_body or len(report_body) < 50:
                report_body = cached_content.strip()
            
            if len(report_body) >= 200:
                # 인사 멘트 생성 (기존 로직과 동일)
                print(f"✅ [유사 질문 처리] 리포트 본문은 캐시에서 가져옴 ({len(report_body)}자), 인사 멘트는 새로 생성")
                
                # final_report_generation과 동일한 방식으로 인사 멘트 생성
                greeting_model_config = {
                    "model": configurable.final_report_model,
                    "max_tokens": configurable.final_report_model_max_tokens,
                    "api_key": get_api_key_for_model(configurable.final_report_model, config),
                }
                
                messages_context = get_buffer_string(messages) if messages else last_user_message
                
                greeting_prompt = f"""당신은 코딩 AI 도구 추천 전문가입니다. 사용자 질문에 맞는 자연스럽고 상세한 인사 멘트를 생성하세요.

사용자 메시지:
{messages_context}

**원칙:**
- 사용자의 현재 질문 내용과 의도를 정확히 파악하여 그에 맞는 자연스러운 멘트를 생성
- 질문의 핵심 키워드(팀 규모, 목적, 요구사항, 도메인 등)를 반영
- 질문에 언급된 구체적인 내용(팀 규모, 목적, 요구사항 등)을 반드시 포함
- 자연스럽고 친절한 톤 유지
- 적절한 길이 (40-100자 정도, 너무 짧지 않게)

인사 멘트만 출력하세요 ([GREETING] 태그 없이, 다른 설명 없이):"""
                
                try:
                    greeting_model = configurable_model.with_config(greeting_model_config)
                    greeting_response = await greeting_model.ainvoke([HumanMessage(content=greeting_prompt)])
                    greeting = str(greeting_response.content).strip().strip('"\'`').strip()
                    
                    if not greeting or len(greeting) < 30:
                        greeting = f"네! {last_user_message[:30]}에 대해 조사해드리겠습니다."
                    
                    print(f"✅ [유사 질문 처리] 인사 멘트 생성 완료: '{greeting}'")
                    
                    return Command(
                        goto="__end__",
                        update={"messages": [
                            AIMessage(content=greeting),
                            AIMessage(content=report_body)
                        ]}
                    )
                except Exception as e:
                    print(f"⚠️ [유사 질문 처리] 인사 멘트 생성 실패: {e}")
                    greeting = f"네! {last_user_message[:30]}에 대해 조사해드리겠습니다."
                    return Command(
                        goto="__end__",
                        update={"messages": [
                            AIMessage(content=greeting),
                            AIMessage(content=report_body)
                        ]}
                    )
    
    # 캐시 미스 및 유사 질문도 없음 → 새로 생성
    print(f"⚠️ [캐시 MISS + 유사 질문 없음] 새로 생성 진행")
    
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
        domain=domain,
        is_followup="YES" if is_followup else "NO"
    )
    
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 🚨 주제 관련성 체크 (항상 실행!)
    if not response.is_on_topic:
        print(f"⚠️ [DEBUG] 주제에서 벗어난 질문 감지")
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.off_topic_message)]}
        )
    
    # 명확화 비활성화 시 바로 다음 단계로 (주제 검증 후)
    if not configurable.allow_clarification:
        print(f"✅ [DEBUG] 주제 검증 통과 - 바로 연구 시작")
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "normalized_query": normalized  # 🆕 정규화 정보 저장
            }
        )
    
    # 명확화 필요 여부 체크
    if response.need_clarification:
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "normalized_query": normalized  # 🆕 정규화 정보 저장
            }
        )


async def write_research_brief(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["research_supervisor"]]:
    """연구 계획 수립"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    configurable = Configuration.from_runnable_config(config)
    domain = state.get("domain", "AI 서비스")
    domain_guide = DOMAIN_GUIDES.get(domain, "")
    
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # domain_guide 포맷팅 (transform_messages에서도 사용)
    try:
        formatted_domain_guide_for_research = domain_guide.format(
            date=get_today_str(),
            current_year=get_current_year(),
            current_month_year=get_current_month_year()
        )
    except KeyError:
        formatted_domain_guide_for_research = domain_guide
    
    # Messages 가져오기 및 Follow-up 판단
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    # 이전 도구 추출 (Follow-up인 경우) - 모든 AI 메시지에서 추출
    previous_tools = ""
    if is_followup:
        all_tools = []
        for msg in reversed(messages_list[:-1]):  # 마지막 사용자 메시지 제외
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                content = str(msg.content)
                # 다양한 패턴으로 도구명 추출
                # 패턴 1: 📊 [도구명]
                tools_found = re.findall(r'📊\s+([^\n]+)', content)
                if tools_found:
                    all_tools.extend([t.strip() for t in tools_found])
                # 패턴 2: ## 📊 [도구명]
                tools_found2 = re.findall(r'##\s+📊\s+([^\n]+)', content)
                if tools_found2:
                    all_tools.extend([t.strip() for t in tools_found2])
                # 패턴 3: **1순위: [도구명]**, **2순위: [도구명]**
                tools_found3 = re.findall(r'\*\*[0-9]+순위:\s*([^\*]+)\*\*', content)
                if tools_found3:
                    all_tools.extend([t.strip() for t in tools_found3])
                # 패턴 4: **최종 추천: [도구명]**
                tools_found4 = re.findall(r'\*\*최종 추천:\s*([^\*]+)\*\*', content)
                if tools_found4:
                    all_tools.extend([t.strip() for t in tools_found4])
        
        # 중복 제거하고 순서 유지
        seen = set()
        unique_tools = []
        for tool in all_tools:
            # 도구명 정제 (불필요한 문자 제거)
            tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
            if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                seen.add(tool_clean)
                unique_tools.append(tool_clean)
        
        previous_tools = ", ".join(unique_tools[:10])  # 최대 10개
        print(f"🔍 [DEBUG] write_research_brief - 이전 추천 도구 추출: {previous_tools}")
    
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(messages_list),
        date=get_today_str(),
        current_year=get_current_year(),
        current_month_year=get_current_month_year(),
        domain=domain,
        domain_guide=formatted_domain_guide_for_research,
        is_followup="YES" if is_followup else "NO",
        previous_tools=previous_tools if previous_tools else "없음",
        question_type="comparison"  # 임시값, LLM이 판단한 값으로 대체됨
    )
    
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 질문 유형은 LLM이 스스로 판단 (response.question_type 사용)
    question_type = response.question_type if hasattr(response, 'question_type') else "comparison"
    
    print(f"🔍 [DEBUG] write_research_brief - Messages: {len(messages_list)}개, 질문 순서: {question_number}번째, Follow-up: {is_followup}, 질문유형: {question_type} (LLM 판단), 이전 도구: {previous_tools}")
    
    # 디버깅: Research Brief와 제약 조건 확인
    print(f"🔍 [DEBUG] Research Brief: {response.research_brief[:200]}...")
    print(f"🔍 [DEBUG] Hard Constraints 추출: {response.hard_constraints}")
    
    # 제약 조건을 dict로 변환하여 state에 저장
    constraints = response.hard_constraints.model_dump() if hasattr(response, 'hard_constraints') and response.hard_constraints else {}
    
    # domain_guide도 포맷팅 필요 (current_year 등 포함)
    try:
        formatted_domain_guide = domain_guide.format(
            date=get_today_str(),
            current_year=get_current_year(),
            current_month_year=get_current_month_year()
        )
    except KeyError:
        # 포맷팅 변수가 없으면 그대로 사용
        formatted_domain_guide = domain_guide
    
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        current_year=get_current_year(),
        current_month_year=get_current_month_year(),
        domain=domain,
        domain_guide=formatted_domain_guide,
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "question_type": response.question_type,  # LLM이 판단한 질문 유형 저장
            "constraints": constraints,  # 제약 조건 저장
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor_tools"]]:
    """연구 슈퍼바이저 (연구 계획 및 위임)"""
    
    configurable = Configuration.from_runnable_config(config)
    
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    tools = [ConductResearch, ResearchComplete, think_tool]
    
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["supervisor", "__end__"]]:
    """슈퍼바이저 도구 실행"""
    
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 종료 조건
    exceeded_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_called = any(
        tc["name"] == "ResearchComplete" for tc in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or no_tool_calls or research_complete_called:
        # notes 추출 (모든 ToolMessage에서 추출)
        notes = get_notes_from_tool_calls(supervisor_messages)
        
        # 디버깅: notes 확인
        print(f"🔍 [DEBUG] supervisor_tools 종료 - notes 개수: {len(notes)}")
        print(f"🔍 [DEBUG] notes 내용: {notes[:2] if notes else '없음'}")
        
        # notes가 비어있으면 raw_notes에서 추출 시도
        if not notes:
            raw_notes = state.get("raw_notes", [])
            if raw_notes:
                print(f"🔍 [DEBUG] raw_notes에서 notes 추출 시도: {len(raw_notes)}개")
                notes = raw_notes if isinstance(raw_notes, list) else [raw_notes]
        
        return Command(
            goto="__end__",
            update={
                "notes": notes if notes else ["연구 결과가 없습니다."],
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # 도구 실행
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # 모든 tool_calls 처리
    for tc in most_recent_message.tool_calls:
        if tc["name"] == "think_tool":
            all_tool_messages.append(ToolMessage(
                content=f"사고 기록: {tc['args']['reflection']}",
                name="think_tool",
                tool_call_id=tc["id"]
            ))
        
        elif tc["name"] == "ConductResearch":
            # 나중에 일괄 처리
            pass
        
        elif tc["name"] == "ResearchComplete":
            all_tool_messages.append(ToolMessage(
                content="연구 완료 확인",
                name="ResearchComplete",
                tool_call_id=tc["id"]
            ))
        
        else:
            # 알 수 없는 tool call에도 응답 (오류 방지)
            all_tool_messages.append(ToolMessage(
                content=f"도구 '{tc['name']}'는 지원되지 않습니다.",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
    
    # ConductResearch 일괄 처리
    conduct_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
    
    if conduct_calls:
        # researcher_subgraph import (순환 참조 방지)
        from app.agent.graph import researcher_subgraph
        
        allowed_calls = conduct_calls[:configurable.max_concurrent_research_units]
        skipped_calls = conduct_calls[configurable.max_concurrent_research_units:]
        
        # 병렬 연구 실행
        tasks = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [HumanMessage(content=tc["args"]["research_topic"])],
                "research_topic": tc["args"]["research_topic"],
                "domain": state.get("domain")
            }, config)
            for tc in allowed_calls
        ]
        
        results = await asyncio.gather(*tasks)
        
        for observation, tc in zip(results, allowed_calls):
            all_tool_messages.append(ToolMessage(
                content=observation.get("compressed_research", "연구 실패"),
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
        
        # 제한 초과로 건너뛴 호출에도 응답 (오류 방지)
        for tc in skipped_calls:
            all_tool_messages.append(ToolMessage(
                content="병렬 연구 제한으로 다음 반복에서 처리됩니다.",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
        
        # raw_notes 수집
        raw_notes_list = []
        for obs in results:
            obs_raw_notes = obs.get("raw_notes", [])
            if obs_raw_notes:
                if isinstance(obs_raw_notes, list):
                    raw_notes_list.extend(obs_raw_notes)
                else:
                    raw_notes_list.append(str(obs_raw_notes))
        
        if raw_notes_list:
            update_payload["raw_notes"] = raw_notes_list
            print(f"🔍 [DEBUG] raw_notes 수집: {len(raw_notes_list)}개")
    
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


async def researcher(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher_tools"]]:
    """개별 연구원 (Vector DB 조회 → 웹 검색)"""
    
    configurable = Configuration.from_runnable_config(config)
    domain = state.get("domain", "AI 서비스")
    domain_guide = DOMAIN_GUIDES.get(domain, "")
    
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    # ========== 🆕 Vector DB 검색 도구 추가 ==========
    async def vector_search(query: str) -> str:
        """Vector DB에서 Facts 검색 (웹 검색 전 우선 시도, threshold 완화)"""
        # threshold를 0.75 → 0.65로 낮춰서 더 많은 결과 가져오기
        facts = vector_store.search_facts(query, limit=5, score_threshold=0.65)
        
        if not facts:
            return "Vector DB에 관련 정보가 없습니다. 웹 검색이 필요합니다."
        
        # 결과가 3개 이상이면 충분하다고 판단
        if len(facts) >= 3:
            formatted = f"✅ Vector DB에서 {len(facts)}개 관련 정보 발견 (충분함):\n\n"
            for idx, fact in enumerate(facts, 1):
                age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
                formatted += f"   {fact['text'][:300]}...\n"
                formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
            return formatted
        
        # 결과가 부족하면 웹 검색 필요
        formatted = f"⚠️ Vector DB에서 {len(facts)}개 관련 정보 발견 (부족함, 웹 검색 필요):\n\n"
        for idx, fact in enumerate(facts, 1):
            age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
            formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
            formatted += f"   {fact['text'][:300]}...\n"
            formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
        formatted += "추가 정보가 필요합니다. 웹 검색을 사용해주세요."
        return formatted
    
    # 검색 도구 정의
    async def web_search(query: str) -> str:
        """웹 검색 도구 (Vector DB에 정보가 없을 때 사용)"""
        result = await searcher.search(
            query=query,
            max_results=configurable.search_max_results,
            search_depth=configurable.search_depth
        )
        
        if not result["success"]:
            return f"검색 실패: {result.get('error', '알 수 없는 오류')}"
        
        # ========== 🆕 검색 결과를 Vector DB에 저장 ==========
        facts_to_store = []
        for r in result["results"]:
            facts_to_store.append({
                "text": f"{r['title']}: {r['content']}",
                "source": result['source'],
                "url": r['url'],
                "metadata": {
                    "score": r.get('score', 0),
                    "query": query
                }
            })
        
        if facts_to_store:
            vector_store.add_facts(facts_to_store, ttl_days=30)
        
        # 결과 포맷팅
        formatted = f"검색 결과 ({result['source']}):\n\n"
        for idx, r in enumerate(result["results"], 1):
            formatted += f"{idx}. {r['title']}\n"
            formatted += f"   URL: {r['url']}\n"
            formatted += f"   내용: {r['content'][:200]}...\n\n"
        
        return formatted
    
    tools = [vector_search, web_search, think_tool]
    
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # domain_guide도 포맷팅 필요 (current_year 등 포함)
    try:
        formatted_domain_guide_researcher = domain_guide.format(
            date=get_today_str(),
            current_year=get_current_year(),
            current_month_year=get_current_month_year()
        )
    except KeyError:
        # 포맷팅 변수가 없으면 그대로 사용
        formatted_domain_guide_researcher = domain_guide
    
    researcher_prompt = research_system_prompt.format(
        domain=domain,
        domain_guide=formatted_domain_guide_researcher,
        date=get_today_str(),
        current_year=get_current_year(),
        current_month_year=get_current_month_year()
    )
    
    messages = [SystemMessage(content=researcher_prompt)] + state.get("researcher_messages", [])
    response = await research_model.ainvoke(messages)
    
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """연구원 도구 실행"""
    
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # 도구 호출 없으면 종료
    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")
    
    # 도구 실행
    tool_outputs = []
    
    for tc in most_recent_message.tool_calls:
        # ========== 🆕 Vector DB 검색 처리 ==========
        if tc["name"] == "vector_search":
            # threshold를 0.75 → 0.65로 낮춰서 더 많은 결과 가져오기
            facts = vector_store.search_facts(tc["args"]["query"], limit=5, score_threshold=0.65)
            
            if facts:
                # 결과가 3개 이상이면 충분하다고 판단
                if len(facts) >= 3:
                    formatted = f"✅ Vector DB에서 {len(facts)}개 관련 정보 발견 (충분함):\n\n"
                    for idx, fact in enumerate(facts, 1):
                        from datetime import datetime
                        age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                        formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
                        formatted += f"   {fact['text'][:300]}...\n"
                        formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
                    content = formatted
                else:
                    # 결과가 부족하면 웹 검색 필요
                    formatted = f"⚠️ Vector DB에서 {len(facts)}개 관련 정보 발견 (부족함, 웹 검색 필요):\n\n"
                    for idx, fact in enumerate(facts, 1):
                        from datetime import datetime
                        age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                        formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
                        formatted += f"   {fact['text'][:300]}...\n"
                        formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
                    formatted += "추가 정보가 필요합니다. 웹 검색을 사용해주세요."
                    content = formatted
            else:
                content = "Vector DB에 관련 정보가 없습니다. 웹 검색을 사용해주세요."
            
            tool_outputs.append(ToolMessage(
                content=content,
                name="vector_search",
                tool_call_id=tc["id"]
            ))
        
        elif tc["name"] == "web_search":
            # 교차 검증 활성화 (Tavily + Serper Fallback)
            result = await searcher.search(
                query=tc["args"]["query"],
                max_results=configurable.search_max_results,
                enable_verification=True  # 교차 검증 활성화
            )
            
            if result["success"]:
                # ========== 🆕 웹 검색 결과를 Vector DB에 저장 ==========
                facts_to_store = []
                for r in result["results"]:
                    facts_to_store.append({
                        "text": f"{r['title']}: {r['content']}",
                        "source": result['source'],
                        "url": r['url'],
                        "metadata": {
                            "score": r.get('score', 0),
                            "query": tc["args"]["query"],
                            "is_official": r.get('is_official', False)
                        }
                    })
                
                if facts_to_store:
                    vector_store.add_facts(facts_to_store, ttl_days=30)
                
                source_info = result.get("source", "unknown")
                if source_info == "verified":
                    verified_info = f"교차 검증됨 (Tavily: {result.get('tavily_count', 0)}개, DuckDuckGo: {result.get('ddg_count', 0)}개 → {result.get('verified_count', 0)}개 검증)"
                else:
                    verified_info = f"({source_info})"
                
                formatted = f"검색 결과 {verified_info}:\n\n"
                
                # 공식 사이트 결과 표시
                official_results = [r for r in result["results"] if r.get("is_official", False)]
                if official_results:
                    formatted += "📌 공식 사이트 결과:\n"
                    for idx, r in enumerate(official_results, 1):
                        formatted += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['content'][:200]}...\n\n"
                
                # 일반 결과
                other_results = [r for r in result["results"] if not r.get("is_official", False)]
                if other_results:
                    if official_results:
                        formatted += "기타 결과:\n"
                    for idx, r in enumerate(other_results, len(official_results) + 1):
                        formatted += f"{idx}. {r['title']}\n   URL: {r['url']}\n   {r['content'][:200]}...\n\n"
                
                # 가격 정보 추출 및 표시 (가격 관련 쿼리인 경우)
                if any(kw in tc["args"]["query"].lower() for kw in ["pricing", "cost", "subscription", "plan", "가격"]):
                    pricing_info = searcher.extract_pricing_info(result["results"])
                    if pricing_info["pricing"]:
                        formatted += f"\n💰 추출된 가격 정보 (신뢰도: {pricing_info['confidence']}):\n"
                        for p in pricing_info["pricing"]:
                            formatted += f"- {p['plan']}: {p['price']} (출처: {len(p['sources'])}개, 공식: {p['official_count']}개)\n"
                
                content = formatted
            else:
                content = f"검색 실패: {result.get('error', '알 수 없는 오류')}"
            
            tool_outputs.append(ToolMessage(
                content=content,
                name="web_search",
                tool_call_id=tc["id"]
            ))
        
        elif tc["name"] == "think_tool":
            tool_outputs.append(ToolMessage(
                content=f"사고: {tc['args']['reflection']}",
                name="think_tool",
                tool_call_id=tc["id"]
            ))
        
        else:
            # 알 수 없는 tool call에도 응답 (오류 방지)
            tool_outputs.append(ToolMessage(
                content=f"도구 '{tc['name']}'는 지원되지 않습니다.",
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
    
    # 종료 조건
    exceeded = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    
    if exceeded:
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """연구 결과 압축"""
    
    configurable = Configuration.from_runnable_config(config)
    
    compression_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
    })
    
    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    compression_prompt = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages
    
    try:
        response = await compression_model.ainvoke(messages)
        
        raw_notes = "\n".join([
            str(msg.content) for msg in researcher_messages
            if isinstance(msg, (ToolMessage, AIMessage))
        ])
        
        return {
            "compressed_research": str(response.content),
            "raw_notes": [raw_notes]
        }
    
    except Exception as e:
        print(f"❌ 압축 실패: {e}")
        return {
            "compressed_research": "연구 결과 압축 실패",
            "raw_notes": [""]
        }


async def run_decision_engine(state: AgentState, config: RunnableConfig):
    """Decision Engine 실행 (의사결정 질문인 경우)"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    # 🚨 최적화: Decision 질문 여부를 먼저 확인 (빠른 반환)
    question_type = state.get("question_type", "comparison")
    messages_list = state.get("messages", [])
    last_user_message = str(messages_list[-1].content).lower() if messages_list else ""
    
    is_decision_question = (
        question_type in ["decision", "comparison"] or
        any(keyword in last_user_message for keyword in [
            "중 하나만", "하나만", "선택", "어떤 것이", "맞을까", "추천", "어떤 도구", 
            "좋을까", "적합", "최적화", "어떤게", "뭘", "무엇을", "어떤게 좋", "어떤 것이 좋",
            "비교", "vs", "대비", "차이", "어떤게 나은", "더 좋은", "어느게", "최적"
        ]) or
        "어떤 도구가 좋을까요" in last_user_message or
        ("어떤 도구" in last_user_message and "좋" in last_user_message) or
        ("vs" in last_user_message or "대비" in last_user_message) or
        ("최적화" in last_user_message and "도구" in last_user_message)  # 🆕 "최적화된 도구" 패턴
    )
    
    if not is_decision_question:
        # Decision 질문이 아니면 Decision Engine 실행 안 함 (빠른 반환)
        return {}
    
    # 🚨 최적화: 제약 조건이 부족한지 먼저 확인 (LLM 호출 전에)
    constraints = state.get("constraints", {})
    team_size = constraints.get("team_size") if constraints else None
    budget_max = constraints.get("budget_max") if constraints else None
    
    # 메시지에서 팀 규모와 예산 추출 시도 (빠른 확인)
    if not team_size and messages_list:
        last_user_msg = str(messages_list[-1].content)
        team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
        if team_size_match:
            team_size = int(team_size_match.group(1))
    
    if not budget_max and messages_list:
        last_user_msg = str(messages_list[-1].content).lower()
        budget_patterns = [
            r'월\s*\$?\s*(\d+)',
            r'\$?\s*(\d+)\s*까지',
            r'\$?\s*(\d+)\s*가능',
            r'\$?\s*(\d+)\s*이하',
            r'\$?\s*(\d+)\s*이내',
        ]
        for pattern in budget_patterns:
            budget_match = re.search(pattern, last_user_msg)
            if budget_match:
                budget_max = float(budget_match.group(1))
                break
    
    # 제약 조건이 부족하면 빠르게 반환 (tool_facts 추출 안 함)
    has_sufficient_constraints = team_size is not None or budget_max is not None
    if not has_sufficient_constraints:
        print(f"⚡ [Decision Engine] 제약 조건 부족 - 빠른 반환 (team_size: {team_size}, budget_max: {budget_max})")
        return {}  # route_after_research에서 clarify_missing_constraints로 라우팅
    
    # 제약 조건이 충분하면 tool_facts 추출 및 Decision Engine 실행
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)
    tool_facts = state.get("tool_facts", [])
    
    print(f"🔍 [Decision Engine DEBUG] is_decision_question: {is_decision_question}, tool_facts: {len(tool_facts) if tool_facts else 0}개")
    print(f"🔍 [Decision Engine DEBUG] findings 길이: {len(findings) if findings else 0}자")
    
    # Findings가 있으면 tool_facts 추출 시도 (최소 길이 50자로 완화)
    if not tool_facts and findings and len(findings.strip()) >= 50:
        print(f"🔍 [Fact Extractor] Findings에서 도구 사실 추출 시작 (Findings 길이: {len(findings)}자)")
        try:
            extracted_facts = await extract_tool_facts(findings, config, max_retries=3)
            if extracted_facts:
                tool_facts = [fact.model_dump() for fact in extracted_facts]
                print(f"✅ [Fact Extractor] {len(tool_facts)}개 도구 사실 추출 완료")
                state["tool_facts"] = tool_facts
            else:
                print(f"⚠️ [Fact Extractor] 도구 사실 추출 실패 (Findings 길이: {len(findings)}자)")
        except Exception as e:
            print(f"⚠️ [Fact Extractor] 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 🚨 Decision 질문인데 tool_facts가 없으면 Findings에서 다시 추출 시도 (더 적극적으로)
    if is_decision_question and not tool_facts:
        if findings and len(findings.strip()) >= 50:
            print(f"🔍 [Decision Engine] tool_facts 없음 - Findings에서 재추출 시도 (Findings 길이: {len(findings)}자)")
            try:
                # 재시도 시 더 긴 max_tokens로 시도 (더 많은 컨텍스트 활용)
                extracted_facts = await extract_tool_facts(findings, config, max_retries=3)
                if extracted_facts:
                    tool_facts = [fact.model_dump() for fact in extracted_facts]
                    print(f"✅ [Decision Engine] 재추출 성공: {len(tool_facts)}개 도구 사실")
                    state["tool_facts"] = tool_facts
                else:
                    print(f"⚠️ [Decision Engine] tool_facts 추출 실패 - Findings에서 도구 정보를 찾을 수 없음 (Findings 길이: {len(findings)}자)")
                    print(f"🔍 [Decision Engine] Findings 샘플 (처음 500자): {findings[:500]}")
            except Exception as e:
                print(f"⚠️ [Decision Engine] tool_facts 추출 오류: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ [Decision Engine] findings가 부족함 ({len(findings) if findings else 0}자, 최소 50자 필요)")
    
    if not tool_facts:
        # Decision 질문인데 tool_facts가 없으면 Decision Engine 실행 불가
        print(f"🚨 [Decision Engine] Decision 질문이지만 tool_facts 없음 - Decision Engine 실행 불가")
        # 🚨 중요: tool_facts가 없으면 decision_result도 없으므로 route_after_research에서 cannot_answer로 감
        # 하지만 사용자가 일반 리포트를 원할 수 있으므로, 빈 dict 반환하여 route_after_research에서 처리하도록 함
        return {}
    
    # Decision Engine 실행
    try:
        constraints = state.get("constraints", {})
        tech_stack = constraints.get("must_support_language", []) if constraints else []
        
        if not tech_stack and messages_list:
            # HumanMessage만 찾아서 사용자 메시지 확인
            human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
            if human_messages:
                last_user_msg = str(human_messages[-1].content).lower()
            else:
                last_user_msg = ""
            # 프로그래밍 언어 추출 (다양한 패턴 인식, 더 유연하게)
            # 백엔드/프론트엔드 키워드에서 스택 추출 (추측적이지만 유용한 정보)
            if "백엔드" in last_user_msg or "backend" in last_user_msg:
                if "java" not in [lang.lower() for lang in tech_stack]:
                    tech_stack.append("Java")
            if "프론트엔드" in last_user_msg or "frontend" in last_user_msg or "프론트" in last_user_msg:
                if "javascript" not in [lang.lower() for lang in tech_stack]:
                    tech_stack.append("JavaScript")
                if "typescript" not in [lang.lower() for lang in tech_stack]:
                    tech_stack.append("TypeScript")
            
            # 일반적인 프로그래밍 언어 키워드 매칭 (더 많은 언어 지원)
            language_keywords_map = {
                "python": "Python",
                "java": "Java",
                "javascript": "JavaScript",
                "js": "JavaScript",
                "typescript": "TypeScript",
                "ts": "TypeScript",
                "go": "Go",
                "golang": "Go",
                "rust": "Rust",
                "c++": "C++",
                "cpp": "C++",
                "c#": "C#",
                "csharp": "C#",
                "php": "PHP",
                "ruby": "Ruby",
                "swift": "Swift",
                "kotlin": "Kotlin",
                "scala": "Scala",
                "node.js": "JavaScript",
                "nodejs": "JavaScript",
                "node": "JavaScript",
                "dart": "Dart",
                "flutter": "Dart",
                "r": "R",
                "matlab": "MATLAB",
                "perl": "Perl",
                "lua": "Lua"
            }
            
            for lang_keyword, lang_name in language_keywords_map.items():
                # 단어 경계를 고려한 매칭 (더 정확하게)
                pattern = r'\b' + re.escape(lang_keyword) + r'\b'
                if re.search(pattern, last_user_msg, re.IGNORECASE):
                    if lang_name not in tech_stack:
                        tech_stack.append(lang_name)
            
            # 프레임워크/라이브러리 키워드에서 언어 추론
            if "react" in last_user_msg or "vue" in last_user_msg or "angular" in last_user_msg:
                if "JavaScript" not in tech_stack:
                    tech_stack.append("JavaScript")
                if "TypeScript" not in tech_stack:
                    tech_stack.append("TypeScript")
            if "spring" in last_user_msg:
                if "Java" not in tech_stack:
                    tech_stack.append("Java")
            if "django" in last_user_msg or "flask" in last_user_msg:
                if "Python" not in tech_stack:
                    tech_stack.append("Python")
        
        # workflow_focus 추출 (모든 사용자 메시지에서 확인)
        workflow_focus = []
        if messages_list:
            # 모든 HumanMessage에서 키워드 확인 (최신 메시지 우선)
            human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
            if human_messages:
                # 모든 사용자 메시지를 합쳐서 확인 (최신 메시지가 우선이지만 이전 맥락도 참고)
                all_user_text = " ".join([str(msg.content).lower() for msg in human_messages])
                last_user_msg = str(human_messages[-1].content).lower()
                
                # 코드 리뷰 요구사항 확인 (더 포괄적이고 유연하게)
                review_keywords = [
                    "pr 리뷰", "pull request 리뷰", "pull request", "pr",
                    "코드 리뷰", "리뷰 지원", "리뷰까지", "리뷰 기능", 
                    "pr 분석", "pr 자동", "코드 작성과 리뷰", "리뷰",
                    "code review", "review", "pullrequest"
                ]
                if any(keyword in all_user_text for keyword in review_keywords):
                    if WorkflowType.CODE_REVIEW not in workflow_focus:
                        workflow_focus.append(WorkflowType.CODE_REVIEW)
                
                # 코드 작성 요구사항 확인 (더 포괄적이고 유연하게)
                code_keywords = [
                    "코드 작성", "코드 생성", "자동완성", "코드 완성", 
                    "코드 작성과 리뷰", "코드", "코딩", "프로그래밍",
                    "code generation", "code completion", "autocomplete",
                    "coding", "programming", "ai assistant", "ai 도구"
                ]
                if any(keyword in all_user_text for keyword in code_keywords):
                    # CODE_GENERATION과 CODE_COMPLETION 모두 추가
                    if WorkflowType.CODE_GENERATION not in workflow_focus:
                        workflow_focus.append(WorkflowType.CODE_GENERATION)
                    if WorkflowType.CODE_COMPLETION not in workflow_focus:
                        workflow_focus.append(WorkflowType.CODE_COMPLETION)
                
                if "리팩토링" in all_user_text:
                    if WorkflowType.REFACTORING not in workflow_focus:
                        workflow_focus.append(WorkflowType.REFACTORING)
                if "디버깅" in all_user_text:
                    if WorkflowType.DEBUGGING not in workflow_focus:
                        workflow_focus.append(WorkflowType.DEBUGGING)
            
            # 기본값: workflow_focus가 비어있으면 CODE_COMPLETION 추가 (일반적인 사용 시나리오)
            # 하지만 이건 선택적이므로, 사용자가 명확히 언급하지 않았으면 빈 리스트도 허용
            # (점수 계산에서 기본값으로 처리됨)
            if not workflow_focus:
                # 사용자가 명시적으로 언급하지 않았으면 기본값 사용하지 않음
                # 점수 계산에서 workflow_focus가 비어있으면 높은 점수 부여하도록 되어 있음
                pass
        
        # UserContext 생성
        current_team_size = None
        current_budget_max = None
        current_required_integrations = []
        
        if messages_list:
            # HumanMessage만 찾아서 사용자 메시지 확인
            human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
            if human_messages:
                all_user_text = " ".join([str(msg.content) for msg in human_messages])
                last_user_msg = all_user_text.lower()
            else:
                last_user_msg = ""
            
            # 팀 규모 추출
            team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
            if team_size_match:
                current_team_size = int(team_size_match.group(1))
            
            # 예산 추출 (월 $XXX, $XXX/월, 월 XXX달러 등)
            budget_patterns = [
                r'월\s*\$?\s*(\d+)',  # "월 $100", "월 100"
                r'\$?\s*(\d+)\s*까지',  # "$100까지", "100까지"
                r'\$?\s*(\d+)\s*가능',  # "$100 가능", "100 가능"
                r'\$?\s*(\d+)\s*이하',  # "$100 이하", "100 이하"
                r'\$?\s*(\d+)\s*이내',  # "$100 이내", "100 이내"
            ]
            for pattern in budget_patterns:
                budget_match = re.search(pattern, last_user_msg)
                if budget_match:
                    current_budget_max = float(budget_match.group(1))
                    break
            
            # 통합 기능 추출 (GitHub, GitLab, Slack 등)
            integration_keywords = {
                "github": "GitHub",
                "gitlab": "GitLab",
                "slack": "Slack",
                "jira": "Jira",
                "bitbucket": "Bitbucket",
                "azure": "Azure DevOps",
                "trello": "Trello",
                "notion": "Notion",
            }
            for keyword, integration_name in integration_keywords.items():
                if keyword in last_user_msg:
                    if integration_name not in current_required_integrations:
                        current_required_integrations.append(integration_name)
        
        # constraints에서 가져온 값이 없으면 메시지에서 추출한 값 사용
        final_team_size = current_team_size or (constraints.get("team_size") if constraints else None)
        final_budget_max = current_budget_max or (constraints.get("budget_max") if constraints else None)
        final_required_integrations = current_required_integrations or (constraints.get("required_integrations", []) if constraints else [])
        
        user_context = UserContext(
            team_size=final_team_size,
            tech_stack=tech_stack,
            budget_max=final_budget_max,
            security_required=constraints.get("security_required", False) if constraints else False,
            required_integrations=final_required_integrations,
            workflow_focus=workflow_focus,
            excluded_tools=constraints.get("excluded_tools", []) if constraints else []
        )
        
        # 🚨 상세 디버깅 로그: 입력 State 출력
        print("=" * 80)
        print("🔍 [Decision Engine INPUT]")
        print(f"  team_size: {final_team_size} (메시지: {current_team_size}, constraints: {constraints.get('team_size') if constraints else None})")
        print(f"  tech_stack: {tech_stack}")
        print(f"  budget_max: {final_budget_max} (메시지: {current_budget_max}, constraints: {constraints.get('budget_max') if constraints else None})")
        print(f"  security_required: {constraints.get('security_required', False) if constraints else False}")
        print(f"  required_integrations: {final_required_integrations} (메시지: {current_required_integrations})")
        print(f"  workflow_focus: {[w.value for w in workflow_focus]}")
        print(f"  excluded_tools: {constraints.get('excluded_tools', []) if constraints else []}")
        print(f"  tool_facts 개수: {len(tool_facts)}개")
        if tool_facts:
            print(f"  tool_facts 도구명: {[fact.get('name', 'Unknown') for fact in tool_facts[:5]]}")
        print("=" * 80)
        
        # Decision Engine 실행
        tools = [ToolFact(**fact) for fact in tool_facts]
        engine = DecisionEngine(user_context)
        decision_result = engine.make_decision(tools)
        
        print(f"✅ [Decision Engine] 실행 완료: 추천 {len(decision_result.recommended_tools)}개, 제외 {len(decision_result.excluded_tools)}개")
        
        return {
            "decision_result": decision_result.model_dump(),
            "tool_facts": tool_facts  # tool_facts를 state에 저장하여 route_after_research에서 사용 가능하도록
        }
    except Exception as e:
        print(f"⚠️ [Decision Engine] 오류: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """최종 리포트 생성 + Redis 캐싱 (일반 리포트, LLM 사용)"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    configurable = Configuration.from_runnable_config(config)
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)
    domain = state.get("domain", "AI 서비스")
    
    # Messages 가져오기 및 Follow-up 판단
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    # 디버깅: findings 확인
    print(f"🔍 [DEBUG] final_report_generation 시작")
    print(f"🔍 [DEBUG] notes 개수: {len(notes)}")
    print(f"🔍 [DEBUG] findings 길이: {len(findings)}자")
    print(f"🔍 [DEBUG] findings 시작 200자: {findings[:200]}")
    print(f"🔍 [DEBUG] is_followup: {is_followup}")
    
    # findings가 비어있을 때 처리
    if not findings or len(findings.strip()) < 50:
        print(f"⚠️ [DEBUG] findings가 비어있거나 너무 짧음: {len(findings)}자")
        
        # Follow-up 질문인 경우 이전 대화 내용 활용
        if is_followup:
            print(f"⚠️ [DEBUG] Follow-up 질문이지만 findings가 비어있음 - 이전 대화 내용 활용")
            # 이전 AI 메시지에서 도구 정보 추출
            previous_ai_messages = [msg for msg in messages_list[:-1] if isinstance(msg, AIMessage)]
            if previous_ai_messages:
                # 마지막 AI 메시지의 내용을 findings로 사용
                last_ai_content = str(previous_ai_messages[-1].content) if previous_ai_messages else ""
                if len(last_ai_content) > 100:
                    findings = f"이전 추천 내용:\n{last_ai_content}\n\n새로운 질문에 대한 추가 분석이 필요합니다."
                    print(f"✅ [DEBUG] 이전 대화 내용을 findings로 사용: {len(findings)}자")
                else:
                    # 이전 대화 내용도 부족하면 research_brief 사용
                    research_brief = state.get("research_brief", "")
                    if research_brief:
                        findings = f"연구 질문: {research_brief}\n\n이전 추천 도구에 대한 추가 정보가 필요합니다."
                    else:
                        findings = "이전에 추천한 도구에 대한 추가 정보를 분석 중입니다."
            else:
                # 이전 대화도 없으면 research_brief 사용
                research_brief = state.get("research_brief", "")
                findings = f"연구 질문: {research_brief}\n\n추가 정보를 분석 중입니다." if research_brief else "추가 정보를 분석 중입니다."
        else:
            # 처음 질문인데 findings가 비어있으면 에러
            error_greeting = "네! 조사해드리겠습니다."
            error_message = "죄송합니다. 연구 결과가 부족하여 답변을 생성할 수 없습니다. 다시 질문해주세요."
            return {
                "final_report": error_message,
                "messages": [
                    AIMessage(content=error_greeting),
                    AIMessage(content=error_message)
                ],
                "notes": {"type": "override", "value": []}
            }
    
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
    }
    
    # Messages 가져오기 및 Follow-up 판단
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    print(f"🔍 [DEBUG] is_followup: {is_followup}, question_number: {question_number}")
    
    # 이전 도구 추출 (Follow-up인 경우) - 모든 AI 메시지에서 추출
    previous_tools = ""
    if is_followup:
        all_tools = []
        for msg in reversed(messages_list[:-1]):  # 마지막 사용자 메시지 제외
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                content = str(msg.content)
                # 다양한 패턴으로 도구명 추출
                # 패턴 1: 📊 [도구명]
                tools_found = re.findall(r'📊\s+([^\n]+)', content)
                if tools_found:
                    all_tools.extend([t.strip() for t in tools_found])
                # 패턴 2: ## 📊 [도구명]
                tools_found2 = re.findall(r'##\s+📊\s+([^\n]+)', content)
                if tools_found2:
                    all_tools.extend([t.strip() for t in tools_found2])
                # 패턴 3: **1순위: [도구명]**, **2순위: [도구명]**
                tools_found3 = re.findall(r'\*\*[0-9]+순위:\s*([^\*]+)\*\*', content)
                if tools_found3:
                    all_tools.extend([t.strip() for t in tools_found3])
                # 패턴 4: **최종 추천: [도구명]**
                tools_found4 = re.findall(r'\*\*최종 추천:\s*([^\*]+)\*\*', content)
                if tools_found4:
                    all_tools.extend([t.strip() for t in tools_found4])
        
        # 중복 제거하고 순서 유지
        seen = set()
        unique_tools = []
        for tool in all_tools:
            # 도구명 정제 (불필요한 문자 제거)
            tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
            if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                seen.add(tool_clean)
                unique_tools.append(tool_clean)
        
        previous_tools = ", ".join(unique_tools[:10])  # 최대 10개
        print(f"🔍 [DEBUG] final_report - 이전 추천 도구 추출: {previous_tools}")
    
    # 질문 유형은 state에서 가져오기 (LLM이 판단한 값)
    question_type = state.get("question_type", "comparison")
    
    print(f"🔍 [DEBUG] final_report - Messages: {len(messages_list)}개, 질문 순서: {question_number}번째, Follow-up: {is_followup}, 질문유형: {question_type} (LLM 판단), 이전 도구: {previous_tools}")
    
    # 제약 조건 가져오기
    constraints = state.get("constraints", {})
    print(f"🔍 [DEBUG] final_report - 제약 조건: {constraints}")
    
    # 제약 조건을 문자열로 포맷팅
    constraints_text = ""
    if constraints:
        constraints_text = "**🚨 하드 제약 조건 (반드시 준수해야 함):**\n\n"
        if constraints.get("budget_max"):
            constraints_text += f"- 최대 예산: {constraints['budget_max']:,}원\n"
        if constraints.get("security_required"):
            constraints_text += f"- 보안/프라이버시: 필수 (외부 서버 전송 금지)\n"
        if constraints.get("excluded_tools"):
            constraints_text += f"- **제외할 도구 (절대 추천 금지)**: {', '.join(constraints['excluded_tools'])}\n"
        if constraints.get("excluded_features"):
            constraints_text += f"- **금지된 기능**: {', '.join(constraints['excluded_features'])}\n"
        if constraints.get("team_size"):
            constraints_text += f"- 팀 규모: {constraints['team_size']}명\n"
        if constraints.get("must_support_ide"):
            constraints_text += f"- 필수 지원 IDE: {', '.join(constraints['must_support_ide'])}\n"
        if constraints.get("must_support_language"):
            constraints_text += f"- 필수 지원 언어: {', '.join(constraints['must_support_language'])}\n"
        if constraints.get("other_requirements"):
            constraints_text += f"- 기타 요구사항: {', '.join(constraints['other_requirements'])}\n"
        constraints_text += "\n**⚠️ 중요**: 위 제약 조건을 위반하는 도구는 추천 목록에서 완전히 제외해야 합니다. 단순히 언급하거나 설명만 하는 것이 아니라, 아예 추천하지 마세요.\n"
    else:
        constraints_text = "제약 조건 없음"
    
    # 🚨 Decision Engine은 run_decision_engine 노드에서 실행되므로 여기서는 실행하지 않음
    # Decision Engine 결과가 있으면 사용, 없으면 일반 리포트 생성 (Discovery 질문용)
    decision_info = ""
    decision_result = state.get("decision_result")
    
    if decision_result:
        # Decision Engine 결과가 있으면 decision_info 생성 (structured_report_generation에서 사용)
        from app.agent.models import DecisionResult
        try:
            result = DecisionResult(**decision_result)
            constraints_dict = state.get("constraints", {})
            team_size = constraints_dict.get("team_size") if constraints_dict else None
            tech_stack = constraints_dict.get("must_support_language", []) if constraints_dict else []
            
            # 비용 분석
            cost_analysis = ""
            tool_facts = state.get("tool_facts", [])
            if team_size and tool_facts:
                for tool_fact_dict in tool_facts:
                    tool_name = tool_fact_dict.get("name", "")
                    if tool_name in result.recommended_tools[:3]:
                        pricing_plans = tool_fact_dict.get("pricing_plans", [])
                        team_plans = [p for p in pricing_plans if p.get("plan_type") in ["team", "business", "enterprise"]]
                        if team_plans:
                            cheapest_plan = min(team_plans, key=lambda p: p.get("price_per_user_per_month") or float('inf'))
                            if cheapest_plan.get("price_per_user_per_month"):
                                monthly_cost = cheapest_plan["price_per_user_per_month"] * team_size
                                annual_cost = monthly_cost * 12
                                cost_analysis += f"- {tool_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년, {team_size}명 기준)\n"
            
            # 상세 점수 분석 (ToolScore 객체를 올바르게 접근)
            detailed_scores = ""
            for score in result.tool_scores:
                # score는 ToolScore 객체이므로 속성으로 직접 접근
                tool_name = score.tool_name
                total_score = score.total_score
                detailed_scores += f"\n**{tool_name}** (내부 평가 점수 참고용, 사용자에게는 자연스럽게 변환):\n"
                detailed_scores += f"  - 언어 지원 점수: {score.language_support_score:.2f} → '언어 지원 우수' 등으로 변환\n"
                detailed_scores += f"  - 통합 기능 점수: {score.integration_score:.2f} → '필요한 통합 지원' 등으로 변환\n"
                detailed_scores += f"  - 업무 적합성 점수: {score.workflow_fit_score:.2f} → '요구사항 적합' 등으로 변환\n"
                detailed_scores += f"  - 가격 점수: {score.price_score:.2f} → '비용 효율적' 등으로 변환\n"
                detailed_scores += f"  - 보안 점수: {score.security_score:.2f} → '보안 요구사항 충족' 등으로 변환\n"
            
            # Decision Engine 결과를 자연스러운 형식으로 변환 (내부 평가 과정 숨김)
            recommended_tools_list = "\n".join([f"{i+1}. {tool}" for i, tool in enumerate(result.recommended_tools[:3])])
            
            decision_info = f"""
**내부 분석 결과 (이 정보를 바탕으로 자연스러운 답변을 생성하세요 - 사용자에게는 내부 평가 과정을 숨기고 자연스러운 추천만 제시하세요!):**

**추천 도구 순서 (위에서부터 우선순위):**
{recommended_tools_list}

**제외된 도구 (추천하지 않아야 할 도구):**
{', '.join(result.excluded_tools) if result.excluded_tools else "없음"}

**비용 정보 ({team_size}명 팀 기준):**
{cost_analysis if cost_analysis else "비용 정보 없음"}

**각 도구별 평가 근거 (내부 참고용, 사용자에게는 자연스럽게 변환):**
{chr(10).join(f"- **{tool}**: {reason}" for tool, reason in result.reasoning.items())}

**🚨 매우 중요: 답변 작성 규칙 (내부 평가 과정을 숨기고 자연스러운 추천만 제시하세요!)**
1. **내부 평가 과정 숨김**: "점수", "Decision Engine", "분석 결과" 같은 내부 마커를 절대 사용하지 마세요!
2. **자연스러운 추천 형식**: "점수가 높아서", "총점 1.0" 같은 표현 대신, "팀 규모와 예산에 가장 적합한", "요구사항을 가장 잘 충족하는" 같은 자연스러운 표현을 사용하세요!
3. **추천 순서 유지**: 위에 나열된 순서대로 추천하되, "1순위", "2순위" 같은 표현 대신 "가장 추천하는 도구", "대안으로 고려할 수 있는 도구" 같은 자연스러운 표현을 사용하세요!
4. **비용 정보 자연스럽게 포함**: 비용 분석을 포함하되, "$XXX/월 (총점 1.0 기준)" 같은 표현 대신, "$XXX/월"만 표시하세요!
5. **판단 이유 자연스럽게 표현**: "언어 지원 점수 1.0" 대신 "언어 지원이 우수하다", "업무 적합성 점수 0.9" 대신 "코드 작성과 리뷰에 적합하다" 같은 자연스러운 표현을 사용하세요!
6. **제외된 도구 처리**: 제외된 도구는 추천하지 않되, "점수 기준으로 제외됨" 같은 표현 대신, "요구사항을 충족하지 않아" 같은 자연스러운 이유를 제시하세요!
7. **코드 리뷰 요구사항 반영**: 사용자가 "코드 작성과 리뷰"를 요청했다면, 추천 도구가 리뷰 기능을 지원하는지 자연스럽게 언급하세요!
8. **명확한 결론**: "둘 다 좋습니다" 같은 중립적 답변 대신, 사용자 상황에 맞는 명확한 추천을 제시하세요!

**⚠️ 절대 금지:**
- "점수", "총점", "Decision Engine", "분석 결과", "평가" 같은 내부 평가 용어 사용 금지!
- "1.0", "0.85" 같은 숫자 점수 직접 노출 금지!
- "🚨🚨🚨 Decision Engine 분석 결과" 같은 내부 마커 사용 금지!
- 사용자가 전문가가 아닌 일반 사용자처럼, 자연스럽고 이해하기 쉬운 답변을 작성하세요!

"""
        except Exception as e:
            print(f"⚠️ [Final Report] DecisionResult 파싱 실패: {e}")
            decision_info = ""
    
    final_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(messages_list),
        findings=findings,
        date=get_today_str(),
        is_followup="YES" if is_followup else "NO",
        previous_tools=previous_tools if previous_tools else "없음",
        question_type=question_type,
        constraints=constraints_text + decision_info
    )
    
    try:
        print(f"🔍 [DEBUG] 리포트 생성 시작 (프롬프트 길이: {len(final_prompt)}자)")
        print(f"🔍 [DEBUG] 프롬프트 시작 300자: {final_prompt[:300]}")
        
        final_report = await configurable_model.with_config(writer_model_config).ainvoke([
            HumanMessage(content=final_prompt)
        ])
        
        print(f"🔍 [DEBUG] 리포트 생성 완료")
        report_content = str(final_report.content).strip()
        print(f"🔍 [DEBUG] 리포트 내용 길이: {len(report_content)}자")
        print(f"🔍 [DEBUG] 리포트 시작 200자: {report_content[:200]}")
        
        # 리포트가 비어있거나 너무 짧으면 에러 처리
        if not report_content or len(report_content) < 50:
            print(f"⚠️ [DEBUG] 리포트가 비어있거나 너무 짧음: {len(report_content)}자")
            print(f"⚠️ [DEBUG] 리포트 전체 내용: {repr(report_content)}")
            error_greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
            error_message = "죄송합니다. 답변을 생성하지 못했습니다. 다시 시도해주세요."
            return {
                "final_report": error_message,
                "messages": [
                    AIMessage(content=error_greeting),
                    AIMessage(content=error_message)
                ],
                "notes": {"type": "override", "value": []}
            }
        
        # ========== 🆕 최종 답변을 Redis에 캐싱 ==========
        normalized_query = state.get("normalized_query", {})
        print(f"🔍 [DEBUG] final_report - normalized_query: {normalized_query}")
        
        if normalized_query and normalized_query.get("cache_key"):
            cache_key = normalized_query["cache_key"]
            print(f"💾 [캐시 저장] 정규화: '{normalized_query.get('normalized_text', '')}' → 캐시키: {cache_key[:16]}...")
            research_cache.set(
                cache_key,
                {"content": report_content},
                domain=domain,
                prefix="final"
            )
            print(f"✅ [캐시 저장] 최종 답변 저장 완료 (캐시키: {cache_key[:16]}..., TTL: 7일)")
            
            # ========== 🆕 질문-캐시 키 매핑을 벡터 DB에 저장 (유사 질문 검색용) ==========
            # 원본 질문 가져오기
            messages_list = state.get("messages", [])
            last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
            
            if last_user_message:
                vector_store.add_query_mapping(
                    query=last_user_message,
                    cache_key=cache_key,
                    normalized_text=normalized_query.get("normalized_text", ""),
                    domain=domain,
                    ttl_days=7
                )
                print(f"✅ [벡터 DB 저장] 질문-캐시 키 매핑 저장 완료 (질문: '{last_user_message[:50]}...')")
        else:
            print(f"⚠️ [캐시 저장 실패] normalized_query 없음: {normalized_query}")
        
        # 마크다운 코드 블록 제거 (```로 시작하고 끝나는 경우)
        report_content = report_content.strip()
        if report_content.startswith("```") and report_content.endswith("```"):
            # 첫 줄의 ``` 제거
            lines = report_content.split('\n')
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # 마지막 줄의 ``` 제거
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            report_content = '\n'.join(lines)
        
        # [GREETING] 태그가 있으면 인사말과 리포트 분리
        print(f"🔍 [DEBUG] 리포트 시작 100자: {report_content[:100]}")
        
        if "[GREETING]" in report_content and "[/GREETING]" in report_content:
            # 태그와 내용을 추출 (여러 줄 포함)
            match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', report_content, re.DOTALL)
            if match:
                greeting = match.group(1).strip()
                # 태그 전체를 제거하고 나머지를 리포트로
                report_body = report_content.replace(match.group(0), "").strip()
                
                print(f"✅ [DEBUG] 인사말 추출 성공: {greeting[:50]}...")
                print(f"✅ [DEBUG] 리포트 본문 길이: {len(report_body)}자")
                print(f"✅ [DEBUG] 리포트 본문 시작: {report_body[:100]}")
                
                # report_body가 비어있으면 원본 report_content 사용
                if not report_body or len(report_body) < 50:
                    print(f"⚠️ [DEBUG] report_body가 비어있음 - 원본 report_content 사용")
                    report_body = report_content
                
                # 두 개의 메시지로 반환
                messages_to_add = [
                    AIMessage(content=greeting),
                    AIMessage(content=report_body)
                ]
            else:
                print(f"❌ [DEBUG] 태그 파싱 실패 - 정규식 매칭 실패")
                # 태그 파싱 실패 시에도 멘트 + 리포트로 분리
                last_msg = messages_list[-1].content.lower() if messages_list else ""
                greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
                messages_to_add = [
                    AIMessage(content=greeting),
                    AIMessage(content=report_content)
                ]
        else:
            print(f"✅ [DEBUG] GREETING 태그 없음 - 키워드 기반 멘트 생성")
            # 키워드 기반으로 간단하고 빠르게 멘트 생성
            last_msg = messages_list[-1].content.lower() if messages_list else ""
            greeting = "네! 조사해드리겠습니다."
            
            if "가격" in last_msg or "얼마" in last_msg or "비용" in last_msg:
                greeting = "네! 가격 정보를 알려드리겠습니다."
            elif "추천" in last_msg or "순위" in last_msg:
                greeting = "네! 조건에 맞춰 추천해드리겠습니다."
            elif "선택" in last_msg or "골라" in last_msg:
                greeting = "네! 최적의 선택을 도와드리겠습니다."
            elif "차이" in last_msg or "비교" in last_msg:
                greeting = "네! 비교 분석해드리겠습니다."
            elif "왜" in last_msg or "이유" in last_msg:
                greeting = "네! 이유를 설명해드리겠습니다."
            elif is_followup:
                greeting = "네! 조건에 맞춰 분석해드리겠습니다."
            
            messages_to_add = [
                AIMessage(content=greeting),
                AIMessage(content=report_content)
            ]
            print(f"✅ [DEBUG] 멘트 생성 완료: '{greeting}'")
        
        return {
            "final_report": report_content,
            "messages": messages_to_add,
            "notes": {"type": "override", "value": []}
        }
    
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 멘트 + 에러 메시지 반환
        error_greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
        error_message = f"죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류: {str(e)}"
        
        return {
            "final_report": error_message,
            "messages": [
                AIMessage(content=error_greeting),
                AIMessage(content=error_message)
            ],
            "notes": {"type": "override", "value": []}
        }


async def structured_report_generation(state: AgentState, config: RunnableConfig):
    """구조화된 리포트 생성 (Decision Engine 결과 기반, 템플릿 사용, LLM 최소화)"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    from app.agent.models import DecisionResult
    
    decision_result_dict = state.get("decision_result")
    if not decision_result_dict:
        # Decision Engine 결과가 없으면 일반 리포트 생성으로 폴백
        return await final_report_generation(state, config)
    
    try:
        decision_result = DecisionResult(**decision_result_dict)
    except Exception as e:
        print(f"⚠️ [Structured Report] DecisionResult 파싱 실패: {e}, 일반 리포트 생성으로 폴백")
        return await final_report_generation(state, config)
    
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    # 사용자 맥락 정보
    constraints = state.get("constraints", {})
    tech_stack = constraints.get("must_support_language", []) if constraints else []
    team_size = constraints.get("team_size") if constraints else None
    
    # 인사 멘트 생성 (간단하게)
    last_user_message = str(messages_list[-1].content) if messages_list else ""
    greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
    
    # LLM을 사용하여 자연스러운 리포트 생성 (내부 평가 과정 완전 숨김)
    # Decision Engine 결과를 기반으로 하지만, LLM이 자연스럽게 변환
    findings = state.get("findings", "")
    notes = state.get("notes", [])
    research_brief = state.get("research_brief", "")
    
    # tool_facts에서 추천 도구 정보 수집
    tool_facts = state.get("tool_facts", [])
    
    # 비용 정보 수집 (검증 로직 포함)
    def get_cost_info(tool_name, team_size):
        for tool_fact_dict in tool_facts:
            if tool_fact_dict.get("name") == tool_name:
                pricing_plans = tool_fact_dict.get("pricing_plans", [])
                if not pricing_plans:
                    continue
                
                # 팀 플랜 우선 검색
                team_plans = [p for p in pricing_plans if p.get("plan_type") in ["team", "business", "enterprise"]]
                if team_plans:
                    cheapest_plan = min(team_plans, key=lambda p: p.get("price_per_user_per_month") or float('inf'))
                    price_per_user = cheapest_plan.get("price_per_user_per_month")
                    if price_per_user and price_per_user > 0:
                        monthly_cost = price_per_user * team_size
                        annual_cost = monthly_cost * 12
                        # 비용이 너무 크면 (예: $1000/월 이상) 검증 필요
                        if monthly_cost > 10000:  # $10,000 이상이면 의심스러움
                            print(f"⚠️ [가격 검증] {tool_name} 계산된 비용이 비정상적으로 큼: ${monthly_cost:.0f}/월 (사용자당 ${price_per_user}/월)")
                        return f"${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                
                # 팀 플랜이 없으면 개인 플랜 확인 (하지만 팀용으로는 추천하지 않음)
                individual_plans = [p for p in pricing_plans if p.get("plan_type") in ["individual", "personal", "pro"]]
                if individual_plans:
                    cheapest_individual = min(individual_plans, key=lambda p: p.get("price_per_user_per_month") or float('inf'))
                    price_per_user = cheapest_individual.get("price_per_user_per_month")
                    if price_per_user and price_per_user > 0:
                        monthly_cost = price_per_user * team_size
                        annual_cost = monthly_cost * 12
                        # 개인 플랜은 참고용으로만 표시 (팀 플랜보다 비쌀 수 있음)
                        return f"개인 플랜 기준: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년, 공식 팀 플랜 확인 권장)"
        return ""
    
    # Decision Engine 결과를 자연스러운 형태로 정리 (내부 평가 용어 완전 제거)
    recommended_tools_info = []
    for i, tool_name in enumerate(decision_result.recommended_tools[:3], 1):
        reasoning_text = decision_result.reasoning.get(tool_name, "")
        # 내부 평가 용어 제거 및 자연스럽게 변환
        reasoning_text = re.sub(r'기술 스택\([^\)]+\)\s*(완벽 지원|부분 지원)', '언어 지원이 우수합니다', reasoning_text)
        reasoning_text = re.sub(r'부분 지원\s*\(\d+%\)', '지원합니다', reasoning_text)
        reasoning_text = re.sub(r'\d+%', '', reasoning_text)
        reasoning_text = re.sub(r'비용 효율적\s*\(\$\d+/월,\s*\$\d+/년\)', '', reasoning_text).strip()
        if not reasoning_text:
            reasoning_text = "팀의 요구사항에 적합한 도구입니다."
        
        cost_info = get_cost_info(tool_name, team_size) if team_size else ""
        recommended_tools_info.append({
            "name": tool_name,
            "reasoning": reasoning_text,
            "cost": cost_info,
            "priority": i
        })
    
    # LLM으로 자연스러운 리포트 생성
    from app.agent.prompts import final_report_generation_prompt, get_today_str
    
    configurable = Configuration.from_runnable_config(config)
    date = get_today_str()
    
    # 제약 조건 텍스트 생성 (간단하게)
    constraints_text_simple = ""
    if constraints:
        if team_size:
            constraints_text_simple += f"팀 규모: {team_size}명\n"
        if tech_stack:
            constraints_text_simple += f"기술 스택: {', '.join(tech_stack)}\n"
        if constraints.get("budget_max"):
            constraints_text_simple += f"예산: 월 ${constraints.get('budget_max')} 이내\n"
    
    # Decision Engine 결과를 자연스러운 형태로 변환 (내부 평가 용어 완전 제거)
    decision_summary = f"""**추천 도구 (우선순위 순서대로):**
{chr(10).join([f"{info['priority']}. {info['name']}: {info['reasoning']}" for info in recommended_tools_info])}

{f"**비용 정보 ({team_size}명 팀 기준):**" + chr(10) + chr(10).join([f"- {info['name']}: {info['cost']}" for info in recommended_tools_info if info['cost']]) if team_size and any(info['cost'] for info in recommended_tools_info) else ""}
"""
    
    # 최종 리포트 생성 프롬프트 (Decision Engine 결과 포함하되 내부 평가 용어 완전 제거)
    combined_constraints = f"{constraints_text_simple}\n\n**내부 분석 결과 (이 정보를 바탕으로 자연스러운 답변을 생성하세요 - 내부 평가 과정은 완전히 숨기세요!):**\n\n{decision_summary}"
    
    report_prompt = final_report_generation_prompt.format(
        research_brief=research_brief,
        messages=get_buffer_string(messages_list[-5:]),
        findings=findings[:3000] if findings else "연구 결과 없음",
        date=date,
        is_followup="YES" if is_followup else "NO",
        previous_tools="",
        question_type="decision",
        constraints=combined_constraints
    )
    
    # LLM으로 리포트 생성
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
    }
    
    try:
        print(f"🔍 [DEBUG] Structured Report 생성 시작 (프롬프트 길이: {len(report_prompt)}자)")
        final_report = await configurable_model.with_config(writer_model_config).ainvoke([
            HumanMessage(content=report_prompt)
        ])
        report_body = str(final_report.content).strip()
        
        # [GREETING] 태그 제거 (final_report_generation과 동일한 로직)
        if "[GREETING]" in report_body and "[/GREETING]" in report_body:
            match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', report_body, re.DOTALL)
            if match:
                report_body = report_body.replace(match.group(0), "").strip()
        
        # 리포트에서 내부 평가 용어 제거 (추가 정리)
        report_body = re.sub(r'🚨🚨🚨\s*Decision Engine.*?🚨🚨🚨', '', report_body, flags=re.DOTALL)
        report_body = re.sub(r'📈\s*상세 점수 분석.*', '', report_body, flags=re.DOTALL)
        report_body = re.sub(r'점수[:\s]*\d+\.?\d*', '', report_body)
        report_body = re.sub(r'총점[:\s]*\d+\.?\d*', '', report_body)
        report_body = re.sub(r'\b보통\b|\b부적합\b|\b부분 지원\b|\b미흡\b|\b미지원\b|\b미충족\b', '', report_body)
        report_body = re.sub(r'\|\s*도구\s*\|\s*언어 지원\s*\|\s*업무 적합성.*?\n', '', report_body, flags=re.DOTALL)  # 비교 테이블 제거
        
        # 리포트 완성도 검증
        if not report_body or len(report_body) < 500:
            print(f"⚠️ [Structured Report] 리포트가 너무 짧음 ({len(report_body)}자) - 재생성 시도")
            raise ValueError("리포트가 너무 짧거나 불완전합니다")
        
        # 추천 도구가 모두 포함되어 있는지 확인
        recommended_count_in_report = sum(1 for tool_name in decision_result.recommended_tools[:3] if tool_name in report_body)
        if recommended_count_in_report < len(decision_result.recommended_tools[:3]):
            print(f"⚠️ [Structured Report] 일부 추천 도구가 리포트에 없음 (포함: {recommended_count_in_report}/{len(decision_result.recommended_tools[:3])}) - 재생성 시도")
            raise ValueError("추천 도구가 모두 포함되지 않았습니다")
        
    except Exception as e:
        print(f"⚠️ [Structured Report] LLM 리포트 생성 실패 또는 불완전: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: 간단하지만 자연스러운 리포트 생성
        report_body = f"## 💡 추천 도구\n\n"
        for info in recommended_tools_info:
            if info['priority'] == 1:
                report_body += f"### 가장 추천하는 도구: {info['name']}\n\n"
            else:
                report_body += f"### 대안 {info['priority']-1}: {info['name']}\n\n"
            
            # reasoning이 있으면 포함, 없으면 간단한 설명 생성
            if info['reasoning'] and len(info['reasoning']) > 10:
                report_body += f"{info['reasoning']}\n\n"
            else:
                report_body += f"{info['name']}은(는) 팀의 요구사항에 적합한 도구입니다. "
                if 'code_completion' in decision_result.recommended_tools and info['name'] == decision_result.recommended_tools[0]:
                    report_body += "코드 작성과 자동 완성 기능을 제공합니다."
                elif 'code_review' in decision_result.recommended_tools and info['name'] in decision_result.recommended_tools:
                    report_body += "코드 리뷰와 품질 검증 기능을 제공합니다."
                report_body += "\n\n"
            
            # 가격 정보 포함 (올바른 계산)
            if info['cost'] and team_size:
                report_body += f"**가격**: {info['cost']}\n\n"
        
        # 모든 추천 도구가 포함되었는지 확인
        if len(report_body) < 500:
            # 추가 정보로 리포트 길이 확보
            report_body += "\n## 💡 결론\n\n"
            if len(recommended_tools_info) > 1:
                report_body += f"위 {len(recommended_tools_info)}개 도구를 조합하여 사용하면 코드 작성과 리뷰 작업을 효율적으로 진행할 수 있습니다.\n\n"
            else:
                report_body += f"{recommended_tools_info[0]['name']}을(를) 사용하면 팀의 개발 생산성을 향상시킬 수 있습니다.\n\n"
    
    # 디버깅: 리포트 생성 결과 확인
    print(f"🔍 [Structured Report DEBUG] 리포트 생성 완료:")
    print(f"  - 추천 도구 개수: {len(decision_result.recommended_tools)}")
    print(f"  - 제외 도구 개수: {len(decision_result.excluded_tools)}")
    print(f"  - 리포트 길이: {len(report_body)}자")
    print(f"  - 리포트 시작 200자: {report_body[:200]}")
    
    # 캐시 저장 (기존 로직과 동일)
    normalized_query = state.get("normalized_query", {})
    domain = state.get("domain", "AI 서비스")
    if normalized_query and normalized_query.get("cache_key"):
        cache_key = normalized_query["cache_key"]
        research_cache.set(
            cache_key,
            {"content": report_body},
            domain=domain,
            prefix="final"
        )
        print(f"✅ [캐시 저장] 구조화된 리포트 저장 완료")
    
    return {
        "final_report": report_body,
        "messages": [
            AIMessage(content=greeting),
            AIMessage(content=report_body)
        ],
        "notes": {"type": "override", "value": []}
    }


async def clarify_missing_constraints(state: AgentState, config: RunnableConfig):
    """제약 조건이 부족할 때 사용자에게 필요한 정보를 질문"""
    
    # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
    import re
    
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    constraints = state.get("constraints", {})
    
    # 부족한 제약 조건 확인
    missing_constraints = []
    
    # 팀 규모 확인
    team_size = constraints.get("team_size") if constraints else None
    if not team_size:
        # 메시지에서 팀 규모 추출 시도
        if messages_list:
            last_user_msg = str(messages_list[-1].content)
            team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
            if not team_size_match:
                missing_constraints.append("팀 규모")
    
    # 예산 확인
    budget_max = constraints.get("budget_max") if constraints else None
    if not budget_max:
        missing_constraints.append("예산 범위")
    
    # 보안 요구사항 확인
    security_required = constraints.get("security_required", False) if constraints else False
    # 보안은 선택사항이므로 필수로 묻지 않음
    
    # 질문 메시지 생성
    if missing_constraints:
        question_parts = []
        if "팀 규모" in missing_constraints:
            question_parts.append("• 몇 명이 사용하시나요? (개인 사용자 / 팀 규모)")
        if "예산 범위" in missing_constraints:
            question_parts.append("• 월 예산 범위는 어느 정도인가요? (무료만 / ~$20 / ~$50 / 무제한)")
        
        question_text = f"""정확한 추천을 위해 다음 정보가 필요합니다:

{chr(10).join(question_parts)}

추가로 다음 정보도 있으면 더 정확한 추천이 가능합니다:
• 코드 외부 전송이 허용되나요? (보안 요구사항)
• 필수로 필요한 통합 기능이 있나요? (예: GitHub, GitLab, Slack 등)
• 주로 어떤 업무에 사용하시나요? (코드 작성, 코드 리뷰, 리팩토링 등)"""
    else:
        # 제약 조건은 있지만 Decision Engine 결과가 없는 경우 (tool_facts 부족 등)
        question_text = "도구 정보가 부족하여 정확한 비교가 어렵습니다. 더 구체적인 정보를 제공해주시면 정확한 추천을 드릴 수 있습니다."
    
    greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
    
    return {
        "final_report": f"{greeting}\n\n{question_text}",
        "messages": [
            AIMessage(content=greeting),
            AIMessage(content=question_text)
        ],
        "notes": {"type": "override", "value": []}
    }


async def cannot_answer(state: AgentState, config: RunnableConfig):
    """Decision Engine 결과 없을 때 답변 불가 메시지 (제약 조건은 충분하지만 tool_facts 부족 등)"""
    
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    greeting = "네! 조건에 맞춰 분석해드리겠습니다." if is_followup else "네! 조사해드리겠습니다."
    error_message = "Decision Engine 분석 결과가 없어 답변할 수 없습니다. 도구 정보가 부족하거나 질문이 명확하지 않을 수 있습니다."
    
    return {
        "final_report": error_message,
        "messages": [
            AIMessage(content=greeting),
            AIMessage(content=error_message)
        ],
        "notes": {"type": "override", "value": []}
    }


def route_after_research(state: AgentState) -> Literal["structured_report_generation", "final_report_generation", "clarify_missing_constraints", "cannot_answer"]:
    """연구 완료 후 라우팅: Decision Engine 결과 유무와 제약 조건 충분 여부에 따라 분기"""
    
    # Decision Engine이 실행되어야 하는 질문인지 확인
    question_type = state.get("question_type", "comparison")
    messages_list = state.get("messages", [])
    last_user_message = str(messages_list[-1].content).lower() if messages_list else ""
    
    # 🚨 디버깅: 질문 내용과 타입 확인
    print(f"🔍 [Routing DEBUG] question_type: {question_type}")
    print(f"🔍 [Routing DEBUG] last_user_message: {last_user_message[:100]}")
    
    is_decision_question = (
        question_type in ["decision", "comparison"] or
        any(keyword in last_user_message for keyword in [
            "중 하나만", "하나만", "선택", "어떤 것이", "맞을까", "추천", "어떤 도구", 
            "좋을까", "적합", "최적화", "어떤게", "뭘", "무엇을", "어떤게 좋", "어떤 것이 좋",
            "비교", "vs", "대비", "차이", "어떤게 나은", "더 좋은", "어느게", "최적"
        ]) or
        "어떤 도구가 좋을까요" in last_user_message or
        ("어떤 도구" in last_user_message and "좋" in last_user_message) or
        ("vs" in last_user_message or "대비" in last_user_message) or
        ("최적화" in last_user_message and "도구" in last_user_message)  # 🆕 "최적화된 도구" 패턴
    )
    
    # 🚨 디버깅: Decision 질문 판정 결과
    print(f"🔍 [Routing DEBUG] is_decision_question: {is_decision_question}")
    
    # Decision Engine 결과 확인
    decision_result = state.get("decision_result")
    tool_facts = state.get("tool_facts", [])
    
    # 제약 조건 충분 여부 확인
    constraints = state.get("constraints", {})
    team_size = constraints.get("team_size") if constraints else None
    budget_max = constraints.get("budget_max") if constraints else None
    
    # 메시지에서 팀 규모 추출 시도
    if not team_size and messages_list:
        last_user_msg = str(messages_list[-1].content)
        import re
        team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
        if team_size_match:
            team_size = int(team_size_match.group(1))
    
    has_sufficient_constraints = team_size is not None or budget_max is not None
    
    # 🚨 디버깅: Decision Engine 결과 확인
    print(f"🔍 [Routing DEBUG] decision_result 존재: {decision_result is not None}")
    print(f"🔍 [Routing DEBUG] decision_result 타입: {type(decision_result)}")
    if decision_result:
        if isinstance(decision_result, dict):
            print(f"🔍 [Routing DEBUG] decision_result.keys(): {list(decision_result.keys())}")
            print(f"🔍 [Routing DEBUG] recommended_tools: {decision_result.get('recommended_tools', [])}")
        else:
            print(f"🔍 [Routing DEBUG] decision_result.recommended_tools: {getattr(decision_result, 'recommended_tools', [])}")
    print(f"🔍 [Routing DEBUG] tool_facts 개수: {len(tool_facts) if tool_facts else 0}")
    print(f"🔍 [Routing DEBUG] 제약 조건 충분 여부: {has_sufficient_constraints} (team_size: {team_size}, budget_max: {budget_max})")
    
    if is_decision_question:
        # 🚨 Decision 질문인 경우
        # Decision Engine 결과가 있고 추천 도구가 있으면 구조화된 리포트 생성
        if decision_result:
            # decision_result가 dict인 경우 model_dump()된 결과이므로 recommended_tools 확인
            if isinstance(decision_result, dict):
                recommended_tools_list = decision_result.get("recommended_tools", [])
            elif hasattr(decision_result, "recommended_tools"):
                recommended_tools_list = decision_result.recommended_tools
            else:
                recommended_tools_list = []
            
            recommended_count = len(recommended_tools_list) if recommended_tools_list else 0
            print(f"🔍 [Routing DEBUG] recommended_count: {recommended_count}")
            
            if recommended_count > 0:
                print(f"✅ [Routing] Decision 질문 + Decision Engine 결과 있음 (추천 {recommended_count}개) → structured_report_generation")
                return "structured_report_generation"
            else:
                # Decision Engine 결과는 있지만 추천 도구가 없는 경우: 필터링이 너무 엄격했을 수 있음
                print(f"⚠️ [Routing DEBUG] Decision Engine 결과는 있지만 추천 도구가 없음 (recommended_tools 빈 리스트)")
                print(f"⚠️ [Routing] 필터링이 너무 엄격했거나 tool_facts 정보 부족 → final_report_generation (fallback)")
                return "final_report_generation"
        elif not has_sufficient_constraints:
            # 제약 조건이 부족하면 사용자에게 질문
            print(f"🔍 [Routing] Decision 질문이지만 제약 조건 부족 → clarify_missing_constraints")
            return "clarify_missing_constraints"
        else:
            # 제약 조건은 충분하지만 Decision Engine 결과가 없음 (tool_facts 부족 등)
            # 🆕 Fallback: 일반 리포트 생성으로 대체 (사용자에게 최소한의 답변 제공)
            print(f"⚠️ [Routing] Decision 질문 + 제약 조건 충분 + Decision Engine 결과 없음 → final_report_generation (fallback)")
            print(f"⚠️ [Routing DEBUG] decision_result: {decision_result}, tool_facts: {len(tool_facts) if tool_facts else 0}개")
            print(f"⚠️ [Routing] tool_facts가 없어 Decision Engine을 실행할 수 없지만, 일반 리포트로 답변 제공")
            return "final_report_generation"
    else:
        # Discovery 질문인 경우: 일반 리포트 생성 (Decision Engine 불필요)
        print(f"✅ [Routing] Discovery 질문 → final_report_generation")
        return "final_report_generation"



