"""연구 계획 수립 노드 - write_research_brief"""

import re
from typing import Literal

from app.agent.nodes._common import (
    Command,
    RunnableConfig,
    AgentState,
    ResearchQuestion,
    Configuration,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    configurable_model,
    DOMAIN_GUIDES,
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    get_today_str,
    get_current_year,
    get_current_month_year,
    get_api_key_for_model,
    AIMessage,
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

