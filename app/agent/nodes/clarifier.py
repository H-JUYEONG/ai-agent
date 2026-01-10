"""명확화 관련 노드 - clarify_missing_constraints, cannot_answer"""

from app.agent.nodes._common import (
    RunnableConfig,
    AgentState,
    AIMessage,
    HumanMessage,
)


async def clarify_missing_constraints(state: AgentState, config: RunnableConfig):
    """제약 조건이 부족할 때 사용자에게 필요한 정보를 질문"""
    
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

