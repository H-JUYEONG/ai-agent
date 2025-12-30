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
            
            # 🚨 캐시 검증: 리포트 본문이 유효한지 확인
            # 리포트가 너무 짧거나(200자 미만) 비어있으면 캐시 무시
            if len(report_body) < 200:
                print(f"⚠️ [캐시 무시] 리포트 본문이 너무 짧음 ({len(report_body)}자). 캐시 무시하고 새로 생성")
                # pass - 캐시를 사용하지 않고 아래 연구 프로세스로 진행
            else:
                # 캐시에서 가져온 답변은 항상 리포트 본문만 있으므로, 멘트를 항상 생성해야 함
                # [GREETING] 태그가 있는지 확인
                greeting_from_cache = ""
                if "[GREETING]" in cached_content and "[/GREETING]" in cached_content:
                    match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', cached_content, re.DOTALL)
                    if match:
                        greeting_from_cache = match.group(1).strip()
                        report_body = cached_content.replace(match.group(0), "").strip()
                        print(f"✅ [캐시] [GREETING] 태그에서 인사말 분리: '{greeting_from_cache[:50]}...'")
                
                # 캐시에 인사말이 있으면 그걸 사용
                if greeting_from_cache:
                    print(f"✅ [캐시 처리] 캐시에서 인사말 발견: '{greeting_from_cache[:50]}...'")
                    return Command(
                        goto="__end__",
                        update={"messages": [
                            AIMessage(content=greeting_from_cache),
                            AIMessage(content=report_body)
                        ]}
                    )
                
                # 🚨 캐시에 인사말이 없으면 생성
                print(f"⚠️ [캐시 처리] 캐시에 인사말 없음 - 멘트 생성")
                
                # 간단한 키워드 기반 멘트 생성 (빠르고 안정적)
                greeting = "네! 조사해드리겠습니다."
                
                if "가격" in last_user_message or "얼마" in last_user_message or "비용" in last_user_message:
                    greeting = "네! 가격 정보를 알려드리겠습니다."
                elif "추천" in last_user_message or "순위" in last_user_message:
                    greeting = "네! 조건에 맞춰 추천해드리겠습니다."
                elif "선택" in last_user_message or "골라" in last_user_message:
                    greeting = "네! 최적의 선택을 도와드리겠습니다."
                elif "차이" in last_user_message or "비교" in last_user_message:
                    greeting = "네! 비교 분석해드리겠습니다."
                elif "왜" in last_user_message or "이유" in last_user_message:
                    greeting = "네! 이유를 설명해드리겠습니다."
                elif is_followup:
                    greeting = "네! 조건에 맞춰 분석해드리겠습니다."
                
                print(f"✅ [캐시 처리] 멘트 생성 완료: '{greeting}'")
                
                return Command(
                    goto="__end__",
                    update={"messages": [
                        AIMessage(content=greeting),
                        AIMessage(content=report_body)
                    ]}
                )
    
    print(f"⚠️ [캐시 MISS] 정규화된 쿼리: '{normalized['normalized_text']}' (키워드: {normalized['keywords']})")
    
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
        """Vector DB에서 Facts 검색 (웹 검색 전 우선 시도)"""
        facts = vector_store.search_facts(query, limit=5, score_threshold=0.75)
        
        if not facts:
            return "Vector DB에 관련 정보가 없습니다. 웹 검색이 필요합니다."
        
        # 결과 포맷팅
        formatted = f"✅ Vector DB에서 {len(facts)}개 관련 정보 발견:\n\n"
        for idx, fact in enumerate(facts, 1):
            age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
            formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
            formatted += f"   {fact['text'][:300]}...\n"
            formatted += f"   출처: {fact['source']} ({fact['url'][:50]}...)\n\n"
        
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
            facts = vector_store.search_facts(tc["args"]["query"], limit=5, score_threshold=0.75)
            
            if facts:
                formatted = f"✅ Vector DB에서 {len(facts)}개 관련 정보 발견:\n\n"
                for idx, fact in enumerate(facts, 1):
                    from datetime import datetime
                    age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                    formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
                    formatted += f"   {fact['text'][:300]}...\n"
                    formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
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
    
    final_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(messages_list),
        findings=findings,
        date=get_today_str(),
        is_followup="YES" if is_followup else "NO",
        previous_tools=previous_tools if previous_tools else "없음",
        question_type=question_type,
        constraints=constraints_text
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



