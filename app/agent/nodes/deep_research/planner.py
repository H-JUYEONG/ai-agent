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
    
    # 이전 도구 추출 (Follow-up인 경우) - 모든 AI 메시지에서 추출 (순서 유지)
    previous_tools = ""
    previous_tools_ordered = []  # 순서 유지용 리스트
    if is_followup:
        all_tools = []
        tools_with_order = []  # 순서 정보 포함
        
        for msg in reversed(messages_list[:-1]):  # 마지막 사용자 메시지 제외
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                content = str(msg.content)
                # 다양한 패턴으로 도구명 추출 (순서 정보 포함)
                # 패턴 1: 📊 [도구명] (순서대로 나타나는 순서 사용)
                tools_found = re.findall(r'📊\s+([^\n]+)', content)
                for idx, tool in enumerate(tools_found):
                    if tool.strip():
                        tools_with_order.append((tool.strip(), idx, "emoji"))
                # 패턴 2: ## 📊 [도구명]
                tools_found2 = re.findall(r'##\s+📊\s+([^\n]+)', content)
                for idx, tool in enumerate(tools_found2):
                    if tool.strip():
                        tools_with_order.append((tool.strip(), idx, "header"))
                # 패턴 3: **1순위: [도구명]**, **2순위: [도구명]** (순서 정보 명시)
                tools_found3 = re.findall(r'\*\*([0-9]+)순위:\s*([^\*]+)\*\*', content)
                for order_str, tool in tools_found3:
                    if tool.strip():
                        order = int(order_str) if order_str.isdigit() else 999
                        tools_with_order.append((tool.strip(), order, "rank"))
                # 패턴 4: **최종 추천: [도구명]**
                tools_found4 = re.findall(r'\*\*최종 추천:\s*([^\*]+)\*\*', content)
                for idx, tool in enumerate(tools_found4):
                    if tool.strip():
                        tools_with_order.append((tool.strip(), 0, "final"))
                # 패턴 5: "가장 추천하는 도구: [도구명]" 또는 "추천하는 도구: [도구명]"
                tools_found5 = re.findall(r'(?:가장\s+)?추천하는\s+도구:\s*([^\n\.]+)', content)
                for idx, tool in enumerate(tools_found5):
                    if tool.strip():
                        # 불필요한 문자 제거 (괄호, 기타 특수문자)
                        tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9/]+$', '', tool.strip()).strip()
                        if tool_clean and len(tool_clean) > 2:
                            tools_with_order.append((tool_clean, 0, "recommended"))
                # 패턴 5-1: "대안 1: [도구명]", "대안 2: [도구명]" 등
                tools_found5_1 = re.findall(r'대안\s*([0-9]+):\s*([^\n\.]+)', content)
                for order_str, tool in tools_found5_1:
                    if tool.strip():
                        order = int(order_str) if order_str.isdigit() else 999
                        # 불필요한 문자 제거 (괄호, 기타 특수문자) - 하지만 도구명 자체는 보존
                        tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9/]+$', '', tool.strip()).strip()
                        # 공백 정리 (여러 공백을 하나로)
                        tool_clean = re.sub(r'\s+', ' ', tool_clean).strip()
                        if tool_clean and len(tool_clean) > 2:
                            tools_with_order.append((tool_clean, order, "alternative"))
                # 패턴 6: "💡 추천 도구" 또는 "💡 맞춤 추천" 섹션의 도구명
                # 💡 섹션에서 도구명 추출 (더 정확한 패턴)
                if "💡" in content and "추천" in content:
                    # 섹션 내에서 도구명 찾기 (더 구체적인 패턴)
                    recommendation_section = re.search(r'💡[^\n]*(?:추천[^\n]*)', content, re.MULTILINE)
                    if recommendation_section:
                        section_content = recommendation_section.group(0)
                        # "가장 추천하는 도구: [도구명]" 패턴 다시 확인
                        tools_found6 = re.findall(r'가장\s+추천하는\s+도구:\s*([^\n\.]+)', section_content)
                        for tool in tools_found6:
                            tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9/]+$', '', tool.strip()).strip()
                            if tool_clean and len(tool_clean) > 2:
                                tools_with_order.append((tool_clean, 0, "recommendation_section"))
                        # GitHub Copilot, Cursor 같은 도구명 패턴 찾기 (섹션 내에서만)
                        tool_names_in_recommendation = re.findall(r'\b(GitHub\s+Copilot|Cursor|Codeium|Tabnine|Aider|Replit|Cline|Windsurf|CodeRabbit|DeepCode|JetBrains\s+AI\s+Assistant|CodeAnt|Qodo|Codacy)\b', section_content, re.IGNORECASE)
                        for tool_name in tool_names_in_recommendation:
                            if tool_name.strip():
                                tools_with_order.append((tool_name.strip(), 999, "recommendation_section"))
        
        # 도구명 정제 및 순서 유지
        seen = set()
        unique_tools = []
        
        # rank 패턴이 있으면 그것을 우선 사용 (순서 정보 명시)
        ranked_tools = [(t, o) for t, o, p in tools_with_order if p == "rank"]
        if ranked_tools:
            ranked_tools.sort(key=lambda x: x[1])  # 순서대로 정렬
            for tool, order in ranked_tools:
                tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
                if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                    seen.add(tool_clean)
                    unique_tools.append(tool_clean)
        
        # alternative 패턴도 순서 정보가 있으므로 우선 처리
        alternative_tools = [(t, o) for t, o, p in tools_with_order if p == "alternative"]
        if alternative_tools:
            alternative_tools.sort(key=lambda x: x[1])  # 순서대로 정렬
            for tool, order in alternative_tools:
                tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
                if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                    seen.add(tool_clean)
                    unique_tools.append(tool_clean)
        
        # 나머지 도구들 추가 (나타난 순서대로)
        for tool, order, pattern in tools_with_order:
            if pattern not in ["rank", "alternative"]:  # 이미 추가된 rank와 alternative는 제외
                tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool).strip()
                if tool_clean and tool_clean not in seen and len(tool_clean) > 2:
                    seen.add(tool_clean)
                    unique_tools.append(tool_clean)
        
        previous_tools_ordered = unique_tools[:10]  # 최대 10개
        previous_tools = ", ".join(previous_tools_ordered)
        print(f"🔍 [DEBUG] write_research_brief - 이전 추천 도구 추출: {previous_tools} (순서 유지)")
    
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
    
    # 이전 추천 도구 순서를 state에 저장 (Follow-up 질문 처리용)
    update_dict = {
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
    
    # Follow-up인 경우 이전 추천 도구 순서 저장
    if is_followup and previous_tools_ordered:
        update_dict["previous_tools_ordered"] = previous_tools_ordered
        print(f"🔍 [DEBUG] 이전 추천 도구 순서 저장: {previous_tools_ordered}")
    
    return Command(
        goto="research_supervisor",
        update=update_dict
    )

