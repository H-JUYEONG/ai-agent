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


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """최종 리포트 생성 + Redis 캐싱"""
    
    configurable = Configuration.from_runnable_config(config)
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)
    domain = state.get("domain", "AI 서비스")
    
    # 🆕 사실 추출 (Findings에서 구조화된 사실 추출)
    tool_facts = state.get("tool_facts", [])
    if not tool_facts and findings and len(findings) > 100:
        print("🔍 [Fact Extractor] Findings에서 도구 사실 추출 시작")
        try:
            extracted_facts = await extract_tool_facts(findings, config)
            if extracted_facts:
                tool_facts = [fact.model_dump() for fact in extracted_facts]
                print(f"✅ [Fact Extractor] {len(tool_facts)}개 도구 사실 추출 완료")
                state["tool_facts"] = tool_facts
        except Exception as e:
            print(f"⚠️ [Fact Extractor] 오류: {e}")
    
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
    
    # Decision Engine 사용 (의사결정 질문인 경우 - 더 넓은 범위로 감지)
    decision_info = ""
    last_user_message = str(messages_list[-1].content).lower() if messages_list else ""
    # 의사결정 질문 감지: decision/comparison 타입이거나, 선택 관련 키워드가 있거나, 추천 요청
    is_decision_question = (
        question_type in ["decision", "comparison"] or  # 비교 질문도 Decision Engine 사용
        any(keyword in last_user_message for keyword in [
            "중 하나만", "하나만", "선택", "어떤 것이", "맞을까", "추천", "어떤 도구", 
            "좋을까", "적합", "어떤게", "뭘", "무엇을", "어떤게 좋", "어떤 것이 좋",
            "비교", "vs", "대비", "차이", "어떤게 나은", "더 좋은", "어느게"
        ]) or
        "어떤 도구가 좋을까요" in last_user_message or
        ("어떤 도구" in last_user_message and "좋" in last_user_message) or
        ("vs" in last_user_message or "대비" in last_user_message)  # 비교 질문도 포함
    )
    
    if is_decision_question:
        try:
            # 사용자 메시지에서 tech_stack 추출
            tech_stack = constraints.get("must_support_language", []) if constraints else []
            if not tech_stack and messages_list:
                last_user_msg = str(messages_list[-1].content).lower()
                
                # 🆕 백엔드/프론트엔드 키워드에서 스택 추출
                if "백엔드" in last_user_msg or "backend" in last_user_msg:
                    # 백엔드 일반적 스택
                    if "java" not in tech_stack:
                        tech_stack.append("Java")
                    if "spring" in last_user_msg or "spring boot" in last_user_msg:
                        # Spring Boot는 Java 기반이므로 이미 추가됨
                        pass
                
                if "프론트엔드" in last_user_msg or "frontend" in last_user_msg or "프론트" in last_user_msg:
                    # 프론트엔드 일반적 스택
                    if "javascript" not in tech_stack:
                        tech_stack.append("JavaScript")
                    if "typescript" not in tech_stack:
                        tech_stack.append("TypeScript")
                    if "react" in last_user_msg:
                        # React는 TypeScript/JavaScript 기반이므로 이미 추가됨
                        pass
                
                # 일반적인 프로그래밍 언어 키워드 매칭
                language_keywords = [
                    "python", "java", "javascript", "typescript", "go", "rust", 
                    "c++", "c#", "php", "ruby", "swift", "kotlin", "scala"
                ]
                for lang_keyword in language_keywords:
                    if lang_keyword in last_user_msg:
                        # 첫 글자 대문자로 변환
                        lang_name = lang_keyword.capitalize()
                        if lang_name == "C++":
                            lang_name = "C++"
                        elif lang_name == "C#":
                            lang_name = "C#"
                        if lang_name not in tech_stack:
                            tech_stack.append(lang_name)
            
            # workflow_focus 추출 (더 정확하게)
            workflow_focus = []
            if messages_list:
                last_user_msg = str(messages_list[-1].content).lower()
                # PR 리뷰 관련 키워드 (우선순위 높음)
                if any(keyword in last_user_msg for keyword in [
                    "pr 리뷰", "pull request 리뷰", "코드 리뷰", "리뷰 지원", 
                    "리뷰까지", "리뷰 기능", "pr 분석", "pr 자동"
                ]):
                    workflow_focus.append(WorkflowType.CODE_REVIEW)
                # 코드 작성 관련
                if any(keyword in last_user_msg for keyword in [
                    "코드 작성", "코드 생성", "자동완성", "코드 완성"
                ]):
                    workflow_focus.append(WorkflowType.CODE_GENERATION)
                # 리팩토링
                if "리팩토링" in last_user_msg:
                    workflow_focus.append(WorkflowType.REFACTORING)
                # 디버깅
                if "디버깅" in last_user_msg:
                    workflow_focus.append(WorkflowType.DEBUGGING)
                # 기본값: workflow_focus가 없으면 코드 작성
                if not workflow_focus:
                    workflow_focus.append(WorkflowType.CODE_COMPLETION)
            
            # UserContext 생성
            # 🆕 제약 조건(팀 규모, 예산 등)은 현재 메시지에서만 추출 (이전 대화의 제약 조건은 사용 안 함)
            current_team_size = None
            if messages_list:
                last_user_msg = str(messages_list[-1].content).lower()
                # 현재 메시지에서 팀 규모 추출
                team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
                if team_size_match:
                    current_team_size = int(team_size_match.group(1))
            
            # 제약 조건은 현재 메시지에서 추출한 값만 사용 (이전 대화의 제약 조건 무시)
            # 단, 이전에 추천한 도구나 언급한 정보는 참조 가능 (이건 constraints가 아니라 previous_tools로 처리)
            team_size = current_team_size  # 현재 메시지에서만 추출
            
            user_context = UserContext(
                team_size=team_size,  # 현재 메시지에서 추출한 값 우선
                tech_stack=tech_stack,
                budget_max=constraints.get("budget_max") if constraints else None,
                security_required=constraints.get("security_required", False) if constraints else False,
                required_integrations=[],  # TODO: 사용자 메시지에서 추출
                workflow_focus=workflow_focus,
                excluded_tools=constraints.get("excluded_tools", []) if constraints else []
            )
            
            # Decision Engine 실행 (도구 사실이 있을 때만)
            tool_facts = state.get("tool_facts", [])
            if tool_facts:
                tools = [ToolFact(**fact) for fact in tool_facts]
                engine = DecisionEngine(user_context)
                decision_result = engine.make_decision(tools)
                
                # 정량적 분석: 비용 계산
                cost_analysis = ""
                if user_context.team_size:
                    for tool in tools:
                        tool_name = tool.name
                        team_plans = [p for p in tool.pricing_plans if p.plan_type in ["team", "business", "enterprise"]]
                        if team_plans:
                            cheapest_plan = min(team_plans, key=lambda p: p.price_per_user_per_month or float('inf'))
                            if cheapest_plan.price_per_user_per_month:
                                monthly_cost = cheapest_plan.price_per_user_per_month * user_context.team_size
                                annual_cost = monthly_cost * 12
                                cost_analysis += f"- {tool_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년, {user_context.team_size}명 기준)\n"
                
                # 상세 점수 분석
                detailed_scores = ""
                for score in decision_result.tool_scores:
                    tool_name = score.tool_name
                    detailed_scores += f"\n**{tool_name}** (총점: {score.total_score:.2f}):\n"
                    detailed_scores += f"  - 언어 지원: {score.language_support_score:.2f}\n"
                    detailed_scores += f"  - 통합 기능: {score.integration_score:.2f}\n"
                    detailed_scores += f"  - 업무 적합성: {score.workflow_fit_score:.2f}\n"
                    detailed_scores += f"  - 가격: {score.price_score:.2f}\n"
                    detailed_scores += f"  - 보안: {score.security_score:.2f}\n"
                
                decision_info = f"""

**🚨🚨🚨 Decision Engine 분석 결과 (반드시 이 결과를 기반으로 답변하세요!) 🚨🚨🚨**

**📊 최종 추천 도구 (점수 순):**
{chr(10).join(f"{i+1}. **{tool}** (총점: {next(s.total_score for s in decision_result.tool_scores if s.tool_name == tool):.2f}/1.0)" for i, tool in enumerate(decision_result.recommended_tools[:3]))}

**❌ 제외된 도구:**
{', '.join(decision_result.excluded_tools) if decision_result.excluded_tools else "없음"}

**💰 정량적 비용 분석 ({user_context.team_size}명 팀 기준):** (팀 규모가 명시된 경우만 표시)
{cost_analysis if cost_analysis else "비용 정보 없음"}

**📈 상세 점수 분석:**
{detailed_scores}

**🎯 판단 이유:**
{chr(10).join(f"- **{tool}**: {reason}" for tool, reason in decision_result.reasoning.items())}

**⚠️⚠️⚠️ 매우 중요: 반드시 위 Decision Engine 결과를 기반으로 답변하세요! ⚠️⚠️⚠️**
1. **추천 도구는 위 순서대로만** 언급하세요. 다른 순서로 나열하지 마세요.
2. **제외된 도구는 절대 추천하지 마세요.** 단순히 언급만 하는 것도 금지입니다.
3. **정량적 비용 분석을 반드시 포함**하세요. 위에 계산된 비용을 그대로 사용하세요.
4. **점수 기반 판단 이유를 명확히 제시**하세요. "점수가 높아서"가 아니라 구체적인 이유를 설명하세요.
5. **명확한 하나의 결론을 제시**하세요. "둘 다 좋습니다" 같은 중립적 답변은 절대 금지입니다.
6. **사용자 스택({', '.join(user_context.tech_stack) if user_context.tech_stack else '전체'}){'과 팀 규모(' + str(user_context.team_size) + '명)' if user_context.team_size else ''}를 기준으로** 판단 이유를 설명하세요. (팀 규모가 명시되지 않았으면 팀 규모 언급 생략)

"""
                # State에 저장
                state["decision_result"] = decision_result.model_dump()
                print(f"✅ [Decision Engine] 실행 완료: 추천 {len(decision_result.recommended_tools)}개, 제외 {len(decision_result.excluded_tools)}개")
        except Exception as e:
            print(f"⚠️ [Decision Engine] 오류: {e}")
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



