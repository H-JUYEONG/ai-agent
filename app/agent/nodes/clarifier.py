"""명확화 관련 노드 - clarify_missing_constraints, cannot_answer"""

from app.agent.nodes._common import (
    RunnableConfig,
    AgentState,
    AIMessage,
    HumanMessage,
)
from app.agent.nodes.writer import generate_greeting_dynamically


async def clarify_missing_constraints(state: AgentState, config: RunnableConfig):
    """제약 조건이 부족할 때 사용자에게 필요한 정보를 질문"""
    
    import re
    
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    constraints = state.get("constraints", {})
    
    # 메시지에서 정보 추출
    last_user_msg = ""
    if messages_list:
        last_user_msg = str(messages_list[-1].content).lower()
    
    # 부족한 제약 조건 확인
    missing_constraints = []
    
    # 팀 규모 확인 (개인 개발자 인식 포함)
    team_size = constraints.get("team_size") if constraints else None
    has_user_type = False
    if not team_size:
        # 메시지에서 팀 규모 추출 시도
        if last_user_msg:
            # "개인", "개인 개발자", "개인 사용자" 등을 인식하여 team_size = 1로 설정
            if any(keyword in last_user_msg for keyword in ["개인", "개인 개발자", "개인 사용자", "개인용", "개인으로"]):
                team_size = 1
                has_user_type = True
            elif any(keyword in last_user_msg for keyword in ["팀", "팀용", "우리 팀", "팀 규모"]):
                has_user_type = True
                # "X명" 패턴 찾기
                team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
                if team_size_match:
                    team_size = int(team_size_match.group(1))
                else:
                    missing_constraints.append("팀 규모")
            else:
                # "X명" 패턴 찾기
                team_size_match = re.search(r'(\d+)\s*명', last_user_msg)
                if not team_size_match:
                    missing_constraints.append("팀 규모")
    
    # 개발 언어/분야 확인
    has_development_area = False
    if last_user_msg:
        # 프로그래밍 언어 확인
        languages = ["python", "javascript", "java", "typescript", "c++", "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "r", "scala", "clojure"]
        # 개발 분야 확인
        domains = ["웹 개발", "백엔드", "프론트엔드", "풀스택", "모바일", "게임", "데이터", "ai", "ml", "머신러닝", "앱 개발"]
        # 프레임워크/라이브러리 확인
        frameworks = ["react", "vue", "angular", "django", "flask", "spring", "node.js", "express", "fastapi", "laravel", "rails"]
        
        # "~으로 개발", "~로 개발", "~ 개발", "~을 사용" 같은 표현도 포함
        if any(lang in last_user_msg for lang in languages) or \
           any(domain in last_user_msg for domain in domains) or \
           any(fw in last_user_msg for fw in frameworks) or \
           re.search(r'으로\s*개발|로\s*개발|개발', last_user_msg):
            has_development_area = True
    
    # 🚨 매우 중요: 사용 형태 + 개발 분야/언어가 모두 있으면 명확화 불필요!
    # (route_after_research에서 이미 확인하므로 여기까지 오지 않아야 함)
    # 하지만 방어적 코딩: 충분한 정보가 있으면 명확화 질문 생성하지 않음
    if has_user_type and has_development_area:
        # 충분한 정보가 있으므로 명확화 질문 생성하지 않음
        # final_report_generation으로 라우팅되도록 빈 메시지 반환
        return {
            "final_report": "",
            "messages": [],
            "notes": {"type": "override", "value": []}
        }
    
    # 예산 확인 (예산 정보가 없어도 추천 가능하므로 선택사항)
    # team_size가 있으면 예산은 필수가 아님
    budget_max = constraints.get("budget_max") if constraints else None
    if not budget_max and not team_size:
        # team_size도 없고 budget_max도 없을 때만 예산 질문
        missing_constraints.append("예산 범위")
    
    # 보안 요구사항 확인
    security_required = constraints.get("security_required", False) if constraints else False
    # 보안은 선택사항이므로 필수로 묻지 않음
    
    # 질문 메시지 생성 (자연스러운 줄글 형식)
    if missing_constraints:
        # 🚨 자연스러운 줄글 형식으로 작성 (리스트/불릿 포인트 절대 금지!)
        question_text = ""
        
        # 개발 분야/언어가 없을 때만 언어/분야 질문
        if not has_development_area:
            if "팀 규모" in missing_constraints and "예산 범위" in missing_constraints:
                question_text = "정확한 추천을 위해 주로 사용하시는 프로그래밍 언어나 개발 분야를 알려주시면 더 맞춤형 추천이 가능합니다. (예: Python, JavaScript, 웹 개발, 백엔드 개발 등)"
            elif "팀 규모" in missing_constraints:
                question_text = "정확한 추천을 위해 주로 사용하시는 프로그래밍 언어나 개발 분야를 알려주시면 더 맞춤형 추천이 가능합니다. (예: Python, JavaScript, 웹 개발, 백엔드 개발 등)"
            elif "예산 범위" in missing_constraints:
                question_text = "더 정확한 추천을 위해 월 예산 범위를 알려주시면 예산에 맞는 도구를 추천해드릴 수 있습니다. (예: 무료만 / ~$20 / ~$50 / 무제한)"
        else:
            # 개발 분야/언어는 있지만 다른 정보가 부족한 경우
            if "팀 규모" in missing_constraints:
                question_text = "정확한 추천을 위해 몇 명이 사용하시는지 알려주시면 더 맞춤형 추천이 가능합니다. (개인 사용자 / 팀 규모)"
            elif "예산 범위" in missing_constraints:
                question_text = "더 정확한 추천을 위해 월 예산 범위를 알려주시면 예산에 맞는 도구를 추천해드릴 수 있습니다. (예: 무료만 / ~$20 / ~$50 / 무제한)"
    else:
        # 제약 조건은 있지만 Decision Engine 결과가 없는 경우 (tool_facts 부족 등)
        question_text = "도구 정보가 부족하여 정확한 비교가 어렵습니다. 더 구체적인 정보를 제공해주시면 정확한 추천을 드릴 수 있습니다."
    
    # LLM으로 동적 멘트 생성
    greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
    if not greeting or len(greeting) < 20:
        # LLM 생성 실패 시 질문 기반 최소 생성
        last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
        if last_user_message:
            greeting = f"{str(last_user_message)[:50]}에 대한 정보가 필요합니다."
        else:
            greeting = "추가 정보가 필요합니다."
        print(f"⚠️ [Clarifier] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
    
    return {
        "final_report": f"{greeting}\n\n{question_text}" if question_text else greeting,
        "messages": [
            AIMessage(content=greeting),
            AIMessage(content=question_text) if question_text else AIMessage(content="")
        ],
        "notes": {"type": "override", "value": []}
    }


async def cannot_answer(state: AgentState, config: RunnableConfig):
    """Decision Engine 결과 없을 때 답변 불가 메시지 (제약 조건은 충분하지만 tool_facts 부족 등)"""
    
    messages_list = state.get("messages", [])
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    question_number = len(human_messages)
    is_followup = question_number > 1
    
    # LLM으로 동적 멘트 생성
    greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
    if not greeting or len(greeting) < 20:
        # LLM 생성 실패 시 질문 기반 최소 생성
        last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
        if last_user_message:
            greeting = f"죄송합니다. {str(last_user_message)[:50]}에 대한 답변을 제공할 수 없습니다."
        else:
            greeting = "죄송합니다. 답변을 제공할 수 없습니다."
        print(f"⚠️ [Cannot Answer] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
    
    error_message = "Decision Engine 분석 결과가 없어 답변할 수 없습니다. 도구 정보가 부족하거나 질문이 명확하지 않을 수 있습니다."
    
    return {
        "final_report": error_message,
        "messages": [
            AIMessage(content=greeting),
            AIMessage(content=error_message)
        ],
        "notes": {"type": "override", "value": []}
    }

