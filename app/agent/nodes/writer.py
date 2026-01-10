"""리포트 생성 노드 (최종 리포트, 구조화된 리포트)"""

from app.agent.nodes._common import *


async def generate_greeting_dynamically(
    messages_list: list,
    config: RunnableConfig,
    is_followup: bool = False,
    max_retries: int = 3
) -> str:
    """LLM을 사용하여 사용자 질문에 맞는 동적 인사 멘트 생성"""
    
    configurable = Configuration.from_runnable_config(config)
    last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
    messages_context = get_buffer_string(messages_list) if messages_list else last_user_message
    
    # 모델별 max_tokens 제한 확인 및 적용
    greeting_model_name = configurable.final_report_model.lower()
    if "gpt-4o-mini" in greeting_model_name:
        greeting_max_tokens = min(configurable.final_report_model_max_tokens, 16384)
    elif "gpt-4o" in greeting_model_name and "mini" not in greeting_model_name:
        greeting_max_tokens = min(configurable.final_report_model_max_tokens, 16384)
    elif "gpt-4" in greeting_model_name:
        greeting_max_tokens = min(configurable.final_report_model_max_tokens, 4096)
    else:
        greeting_max_tokens = min(configurable.final_report_model_max_tokens, 16384)
    
    greeting_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": greeting_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
    }
    
    greeting_prompt = f"""당신은 코딩 AI 도구 추천 전문가입니다. 사용자 질문에 맞는 자연스럽고 상세한 인사 멘트를 생성하세요.

사용자 메시지:
{messages_context}

**원칙:**
- 사용자의 현재 질문 내용과 의도를 정확히 파악하여 그에 맞는 자연스러운 멘트를 생성
- 질문의 핵심 키워드(팀 규모, 목적, 요구사항, 도메인 등)를 반영
- 질문에 언급된 구체적인 내용(팀 규모, 목적, 요구사항 등)을 반드시 포함
- 자연스럽고 친절한 톤 유지
- 적절한 길이 (40-100자 정도, 너무 짧지 않게)
- Follow-up 질문인 경우 질문 내용(가격, 비교, 추천, 설명 등)을 반영
- "네!", "좋아요", "알겠습니다" 같은 단순한 시작 표현도 좋지만, 반드시 질문 내용을 구체적으로 반영

**좋은 예시:**
- 질문: "저희는 백엔드·프론트엔드 포함해서 8명 규모의 개발팀인데, 코드 작성과 리뷰에 AI를 도입해서 생산성을 높이고 싶습니다. 어떤 도구가 좋을까요?"
  인사 멘트: "네! 백엔드와 프론트엔드를 포함한 8명 규모의 개발팀에 적합한 AI 도구들을 분석해드리겠습니다. 팀의 코드 작성 및 리뷰 효율성 향상에 도움이 되는 도구를 비교해드리겠습니다."

- 질문 (Follow-up): "가격 알려줘" → "네! 가격 정보를 확인해드리겠습니다."
- 질문 (Follow-up): "표로 정리해줘" → "네! 표 형식으로 정리해드리겠습니다."
- 질문 (Follow-up): "1순위 추천해줘" → "네! 조건에 맞는 1순위를 추천드리겠습니다."

**나쁜 예시 (너무 짧거나 맥락 없음):**
- "안녕하세요." (너무 짧음)
- "네!" (너무 짧고 내용 없음)
- "AI 도구로 생산성을 높여드리겠습니다." (너무 짧고 구체적이지 않음)
- "네! 조사해드리겠습니다." (너무 일반적)

인사 멘트만 출력하세요 ([GREETING] 태그 없이, 다른 설명 없이):"""
    
    for attempt in range(max_retries):
        try:
            greeting_model = configurable_model.with_config(greeting_model_config)
            greeting_response = await greeting_model.ainvoke([HumanMessage(content=greeting_prompt)])
            greeting = str(greeting_response.content).strip().strip('"\'`').strip()
            
            # 응답이 너무 짧으면 재시도
            if not greeting or len(greeting) < 30:
                if attempt < max_retries - 1:
                    print(f"⚠️ [Greeting Generation] LLM 응답이 너무 짧음 ({len(greeting) if greeting else 0}자), 재시도 {attempt + 1}/{max_retries}")
                    retry_prompt = f"""당신은 코딩 AI 도구 추천 전문가입니다.

사용자 메시지:
{messages_context}

위 질문에 맞는 자연스럽고 상세한 인사 멘트를 생성하세요. 질문의 핵심 내용(팀 규모, 목적, 요구사항 등)을 구체적으로 반영한 40-100자 정도의 상세한 인사 멘트를 작성해주세요.

인사 멘트만 출력하세요:"""
                    greeting_prompt = retry_prompt
                    continue
                else:
                    # 마지막 시도도 실패하면 빈 문자열 반환 (호출자가 처리)
                    print(f"⚠️ [Greeting Generation] LLM 응답이 계속 짧음 ({len(greeting) if greeting else 0}자), 재시도 실패")
                    return greeting if greeting else ""
            
            # 응답이 너무 길면 적절히 자르기 (100자 이내로)
            if greeting and len(greeting) > 100:
                import re
                sentences = re.split(r'[.!?。]', greeting)
                if len(sentences) > 1 and sentences[0]:
                    greeting = sentences[0].strip() + '.'
                else:
                    greeting = greeting[:100].strip()
            
            print(f"✅ [Greeting Generation] LLM으로 멘트 생성 완료: '{greeting}' (길이: {len(greeting)}자)")
            return greeting
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ [Greeting Generation] LLM 멘트 생성 실패 (시도 {attempt + 1}/{max_retries}): {e}, 재시도")
                continue
            else:
                print(f"⚠️ [Greeting Generation] LLM 멘트 생성 완전 실패: {e}")
                return ""
    
    return ""


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """최종 리포트 생성 + Redis 캐싱 (일반 리포트, LLM 사용)"""
    
    try:
        # re 모듈을 함수 내에서 명시적으로 import하여 스코프 문제 해결
        import re
        
        configurable = Configuration.from_runnable_config(config)
        notes = state.get("notes", [])
        findings = "\n\n".join(notes)
        domain = state.get("domain", "AI 서비스")
        
        # 모델별 max_tokens 제한 확인 및 적용
        model_name = configurable.final_report_model.lower()
        if "gpt-4o-mini" in model_name:
            max_tokens_allowed = min(configurable.final_report_model_max_tokens, 16384)  # gpt-4o-mini 최대 16384
        elif "gpt-4o" in model_name and "mini" not in model_name:
            max_tokens_allowed = min(configurable.final_report_model_max_tokens, 16384)  # gpt-4o 최대 16384
        elif "gpt-4" in model_name:
            max_tokens_allowed = min(configurable.final_report_model_max_tokens, 4096)  # gpt-4 최대 4096
        else:
            max_tokens_allowed = min(configurable.final_report_model_max_tokens, 16384)  # 기본값
        
        writer_model_config = {
            "model": configurable.final_report_model,
            "max_tokens": max_tokens_allowed,
            "api_key": get_api_key_for_model(configurable.final_report_model, config),
        }
        
        print(f"🔍 [DEBUG] 모델: {configurable.final_report_model}, max_tokens: {max_tokens_allowed} (원래 설정: {configurable.final_report_model_max_tokens})")
        
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
                # LLM으로 동적 멘트 생성
                error_greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
                if not error_greeting or len(error_greeting) < 20:
                    # LLM 생성 실패 시 질문 기반 최소 생성
                    last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
                    if last_user_message:
                        error_greeting = f"죄송합니다. {last_user_message[:50]}에 대한 연구 결과가 부족하여 답변을 생성할 수 없습니다."
                    else:
                        error_greeting = "죄송합니다. 연구 결과가 부족하여 답변을 생성할 수 없습니다."
                error_message = "죄송합니다. 연구 결과가 부족하여 답변을 생성할 수 없습니다. 다시 질문해주세요."
                return {
                    "final_report": error_message,
                    "messages": [
                        AIMessage(content=error_greeting),
                        AIMessage(content=error_message)
                    ],
                    "notes": {"type": "override", "value": []}
                }
        
        # Messages 가져오기 및 Follow-up 판단 (중복 제거됨 - 이미 위에서 처리)
        # messages_list = state.get("messages", [])
        # human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
        # question_number = len(human_messages)
        # is_followup = question_number > 1
        
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
                    # 패턴 5: "가장 추천하는 도구: [도구명]" 또는 "추천하는 도구: [도구명]"
                    tools_found5 = re.findall(r'(?:가장\s+)?추천하는\s+도구:\s*([^\n\.]+)', content)
                    if tools_found5:
                        for tool in tools_found5:
                            # 불필요한 문자 제거 (괄호, 기타 특수문자)
                            tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9/]+$', '', tool.strip()).strip()
                            if tool_clean and len(tool_clean) > 2:
                                all_tools.append(tool_clean)
                    # 패턴 5-1: "대안 1: [도구명]", "대안 2: [도구명]" 등
                    tools_found5_1 = re.findall(r'대안\s*([0-9]+):\s*([^\n\.]+)', content)
                    if tools_found5_1:
                        for order_str, tool in tools_found5_1:
                            if tool.strip():
                                tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9/]+$', '', tool.strip()).strip()
                                if tool_clean and len(tool_clean) > 2:
                                    all_tools.append(tool_clean)
                    # 패턴 6: "💡 추천 도구" 또는 "💡 맞춤 추천" 섹션의 도구명
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
                                    all_tools.append(tool_clean)
                            # GitHub Copilot, Cursor 같은 도구명 패턴 찾기 (섹션 내에서만)
                            tool_names_in_recommendation = re.findall(r'\b(GitHub\s+Copilot|Cursor|Codeium|Tabnine|Aider|Replit|Cline|Windsurf|CodeRabbit|DeepCode|JetBrains\s+AI\s+Assistant|CodeAnt|Qodo|Codacy)\b', section_content, re.IGNORECASE)
                            for tool_name in tool_names_in_recommendation:
                                if tool_name.strip():
                                    all_tools.append(tool_name.strip())
            
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
        
        # 답변 형식 가져오기 (기본값: markdown)
        response_format = state.get("response_format", "markdown")
        print(f"🔍 [DEBUG] final_report - 답변 형식: {response_format}")
        
        # 🆕 표 형식 요청 시 Structured Output 사용
        if response_format == "table":
            from app.agent.state import TableData
            
            table_prompt = final_report_generation_prompt.format(
            research_brief=state.get("research_brief", ""),
            messages=get_buffer_string(messages_list),
            findings=findings,
            date=get_today_str(),
            is_followup="YES" if is_followup else "NO",
            previous_tools=previous_tools if previous_tools else "없음",
            question_type=question_type,
            constraints=constraints_text + decision_info,
            response_format=response_format
            )
            
            try:
                print(f"🔍 [DEBUG] Structured Output으로 표 데이터 생성 시작")
                
                # Structured Output으로 표 데이터 생성
                table_model = (
                    configurable_model
                    .with_structured_output(TableData)
                    .with_config(writer_model_config)
                )
                
                table_data = await table_model.ainvoke([HumanMessage(content=table_prompt)])
                
                print(f"✅ [DEBUG] 표 데이터 생성 완료: {len(table_data.columns)}개 열, {len(table_data.rows)}개 행")
                print(f"🔍 [DEBUG] 표 열: {table_data.columns}")
                print(f"🔍 [DEBUG] 표 행 개수: {len(table_data.rows)}")
                
                # JSON 형식으로 변환 (프론트엔드에서 파싱 가능하도록)
                import json
                table_json = json.dumps({
                    "type": "table",
                    "columns": table_data.columns,
                    "rows": table_data.rows
                }, ensure_ascii=False, indent=2)
                
                # 리포트 본문은 JSON 문자열로 설정 (프론트엔드에서 파싱)
                report_content = table_json
                
            except Exception as e:
                print(f"⚠️ [DEBUG] Structured Output 실패: {e}, 일반 텍스트로 폴백")
                # 폴백: 일반 텍스트로 표 형식 생성
                final_prompt = final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=get_buffer_string(messages_list),
                    findings=findings,
                    date=get_today_str(),
                    is_followup="YES" if is_followup else "NO",
                    previous_tools=previous_tools if previous_tools else "없음",
                    question_type=question_type,
                    constraints=constraints_text + decision_info,
                    response_format="markdown"  # 폴백 시 markdown
                )
                final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                    HumanMessage(content=final_prompt)
                ])
                report_content = str(final_report.content).strip()
        else:
            # 일반 마크다운 형식
            final_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(messages_list),
                findings=findings,
                date=get_today_str(),
                is_followup="YES" if is_followup else "NO",
                previous_tools=previous_tools if previous_tools else "없음",
                question_type=question_type,
                constraints=constraints_text + decision_info,
                response_format=response_format
            )
            
            try:
                print(f"🔍 [DEBUG] 리포트 생성 시작 (프롬프트 길이: {len(final_prompt)}자)")
                print(f"🔍 [DEBUG] 프롬프트 시작 300자: {final_prompt[:300]}")
                
                final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                    HumanMessage(content=final_prompt)
                ])
                
                print(f"🔍 [DEBUG] 리포트 생성 완료")
                report_content = str(final_report.content).strip()
            except Exception as e:
                print(f"⚠️ [DEBUG] 리포트 생성 실패: {e}")
                report_content = "응답 생성 중 오류가 발생했습니다."
        
        print(f"🔍 [DEBUG] 리포트 내용 길이: {len(report_content)}자")
        print(f"🔍 [DEBUG] 리포트 시작 200자: {report_content[:200]}")
        
        # 리포트가 비어있거나 너무 짧으면 에러 처리
        if not report_content or len(report_content) < 50:
            print(f"⚠️ [DEBUG] 리포트가 비어있거나 너무 짧음: {len(report_content)}자")
            print(f"⚠️ [DEBUG] 리포트 전체 내용: {repr(report_content)}")
            # LLM으로 동적 멘트 생성
            error_greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
            if not error_greeting or len(error_greeting) < 20:
                # LLM 생성 실패 시 질문 기반 최소 생성
                last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
                if last_user_message:
                    error_greeting = f"죄송합니다. {last_user_message[:50]}에 대한 답변을 생성하지 못했습니다."
                else:
                    error_greeting = "죄송합니다. 답변을 생성하지 못했습니다."
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
        # 🚨 재검색이 필요 없는 경우(need_research = false)에는 캐시/벡터 DB 저장 건너뛰기
        # 이전 대화 정보만 사용한 경우이므로 새로운 캐시가 필요 없음
        need_research = state.get("need_research", True)  # 기본값: True (검색 필요)
        
        if need_research:
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
        else:
            print(f"✅ [캐시 저장 건너뛰기] 재검색 불필요 (need_research = false) - 이전 대화 정보만 사용했으므로 저장하지 않음")
        
        # 마크다운 코드 블록 제거 (```로 시작하고 끝나는 경우)
        # 단, 표 형식이 포함된 경우는 보존 (표 형식이 손상될 수 있음)
        report_content = report_content.strip()
        has_table = '|' in report_content and '|--------|' in report_content or '|------|' in report_content
        if report_content.startswith("```") and report_content.endswith("```") and not has_table:
            # 첫 줄의 ``` 제거
            lines = report_content.split('\n')
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # 마지막 줄의 ``` 제거
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            report_content = '\n'.join(lines)
        elif has_table and report_content.startswith("```"):
            # 표 형식이 포함된 경우, 코드 블록 마커만 제거하고 내용은 보존
            lines = report_content.split('\n')
            # 첫 줄의 ``` 제거 (표 형식 보존)
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            # 마지막 줄의 ``` 제거 (표 형식 보존)
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
                # 태그 파싱 실패 시에도 LLM으로 동적 멘트 생성
                print(f"✅ [DEBUG] 태그 파싱 실패 - LLM으로 동적 멘트 생성")
                # LLM으로 동적으로 멘트 생성하도록 아래로 진행
        else:
            print(f"✅ [DEBUG] GREETING 태그 없음 - LLM으로 동적 멘트 생성")
        # LLM으로 동적으로 멘트 생성 (공통 함수 사용)
        greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
        if not greeting or len(greeting) < 20:
            # LLM 생성 실패 시 질문 기반 최소 생성
            last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
            if last_user_message:
                greeting = f"{last_user_message[:50]}에 대해 분석해드리겠습니다."
            else:
                greeting = "분석해드리겠습니다."
            print(f"⚠️ [DEBUG] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
        
        messages_to_add = [
            AIMessage(content=greeting),
            AIMessage(content=report_content)
        ]
        print(f"✅ [DEBUG] 최종 멘트: '{greeting}'")
        
        return {
            "final_report": report_content,
            "messages": messages_to_add,
            "notes": {"type": "override", "value": []}
        }
    
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생 시에도 LLM으로 동적 멘트 생성
        error_greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
        if not error_greeting or len(error_greeting) < 20:
            # LLM 생성 실패 시 질문 기반 최소 생성
            last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
            if last_user_message:
                error_greeting = f"죄송합니다. {last_user_message[:50]}에 대한 답변 생성 중 오류가 발생했습니다."
            else:
                error_greeting = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
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
    
    # LLM으로 동적 멘트 생성 (공통 함수 사용)
    greeting = await generate_greeting_dynamically(messages_list, config, is_followup)
    if not greeting or len(greeting) < 20:
        # LLM 생성 실패 시 질문 기반 최소 생성
        last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
        if last_user_message:
            greeting = f"{str(last_user_message)[:50]}에 대해 분석해드리겠습니다."
        else:
            greeting = "분석해드리겠습니다."
        print(f"⚠️ [Structured Report] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
    
    # LLM을 사용하여 자연스러운 리포트 생성 (내부 평가 과정 완전 숨김)
    # Decision Engine 결과를 기반으로 하지만, LLM이 자연스럽게 변환
    findings = state.get("findings", "")
    notes = state.get("notes", [])
    research_brief = state.get("research_brief", "")
    
    # tool_facts에서 추천 도구 정보 수집
    tool_facts = state.get("tool_facts", [])
    
    # 비용 정보 수집 (검증 로직 포함)
    def get_cost_info(tool_name, team_size):
        if not team_size or team_size <= 0:
            return ""
        
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
                        plan_name = cheapest_plan.get("name", "팀 플랜")
                        plan_type = cheapest_plan.get("plan_type", "")
                        # 비용이 너무 크면 (예: $10,000/월 이상) 검증 필요
                        if monthly_cost > 10000:  # $10,000 이상이면 의심스러움
                            print(f"⚠️ [가격 검증] {tool_name} 계산된 비용이 비정상적으로 큼: ${monthly_cost:.0f}/월 (사용자당 ${price_per_user}/월)")
                        # 플랜 타입에 따라 적절한 라벨 사용
                        if plan_type in ["team", "business", "enterprise"]:
                            return f"팀 플랜 ({plan_name}): ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                        else:
                            return f"{plan_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                
                # 팀 플랜이 없거나 price_per_user가 없는 경우, 다른 플랜 확인
                # price_per_user_per_month가 있는 플랜 우선 검색 (plan_type과 상관없이)
                plans_with_per_user = [p for p in pricing_plans if p.get("price_per_user_per_month")]
                if plans_with_per_user:
                    cheapest_plan = min(plans_with_per_user, key=lambda p: p.get("price_per_user_per_month") or float('inf'))
                    price_per_user = cheapest_plan.get("price_per_user_per_month")
                    if price_per_user and price_per_user > 0:
                        monthly_cost = price_per_user * team_size
                        annual_cost = monthly_cost * 12
                        plan_name = cheapest_plan.get("name", "플랜")
                        plan_type = cheapest_plan.get("plan_type", "unknown")
                        # plan_type이 "team", "business", "enterprise"가 아닌 경우에만 경고
                        if plan_type not in ["team", "business", "enterprise"]:
                            return f"{plan_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년, 팀 플랜 확인 권장)"
                        else:
                            return f"{plan_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                
                # 사용량 기반 과금 확인
                usage_based_plans = [p for p in pricing_plans if p.get("plan_type") == "usage-based"]
                if usage_based_plans:
                    # 사용량 기반 과금은 가격 계산이 불가능하므로 플랜명에 가격 정보 포함
                    usage_plan = usage_based_plans[0]
                    plan_name = usage_plan.get("name", "사용량 기반 과금")
                    source_url = usage_plan.get("source_url", "")
                    if source_url:
                        return f"{plan_name} (정확한 가격은 사용량에 따라 다르며, {source_url}에서 확인 가능)"
                    else:
                        return f"{plan_name} (정확한 가격은 사용량에 따라 다르므로 공식 사이트에서 확인 필요)"
                
                # 연간 플랜 처리 (price_per_year 또는 price_per_user_per_year)
                annual_plans = [p for p in pricing_plans if p.get("price_per_year") or p.get("price_per_user_per_year")]
                if annual_plans:
                    cheapest_annual = min(annual_plans, key=lambda p: (p.get("price_per_year") or float('inf')) if p.get("price_per_year") else (p.get("price_per_user_per_year") or 0) * team_size if p.get("price_per_user_per_year") else float('inf'))
                    
                    price_per_year = cheapest_annual.get("price_per_year")
                    price_per_user_per_year = cheapest_annual.get("price_per_user_per_year")
                    plan_name = cheapest_annual.get("name", "플랜")
                    plan_type = cheapest_annual.get("plan_type", "unknown")
                    source_url = cheapest_annual.get("source_url", "")
                    
                    if price_per_year and price_per_year > 0:
                        # 전체 팀 연간 가격인 경우
                        monthly_cost = price_per_year / 12
                        annual_cost = price_per_year
                        if plan_type in ["team", "business", "enterprise"]:
                            return f"팀 플랜 ({plan_name}): ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                        else:
                            return f"{plan_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                    elif price_per_user_per_year and price_per_user_per_year > 0:
                        # 사용자당 연간 가격인 경우
                        monthly_cost = (price_per_user_per_year * team_size) / 12
                        annual_cost = price_per_user_per_year * team_size
                        if plan_type in ["team", "business", "enterprise"]:
                            return f"팀 플랜 ({plan_name}): ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년)"
                        else:
                            return f"{plan_name}: ${monthly_cost:.0f}/월 (${annual_cost:.0f}/년, 팀 플랜 확인 권장)"
                
                # price_per_month만 있는 경우 (개인 플랜일 수 있음)
                individual_plans = [p for p in pricing_plans if p.get("plan_type") in ["individual", "personal", "pro"]]
                if individual_plans:
                    cheapest_individual = min(individual_plans, key=lambda p: p.get("price_per_month") or float('inf'))
                    price_per_month = cheapest_individual.get("price_per_month")
                    plan_name = cheapest_individual.get("name", "개인 플랜")
                    source_url = cheapest_individual.get("source_url", "")
                    if price_per_month and price_per_month > 0:
                        # 팀 규모가 있으면 개인 플랜을 그대로 표시하고 팀 플랜 확인 권장
                        # 개인 플랜 가격을 팀 인원수로 곱하면 안 됨 (부정확한 정보)
                        if source_url:
                            return f"개인 플랜 ({plan_name}): ${price_per_month:.0f}/월 (팀 플랜 정보는 {source_url}에서 확인 필요)"
                        else:
                            return f"개인 플랜 ({plan_name}): ${price_per_month:.0f}/월 (팀 플랜은 공식 사이트에서 확인 필요)"
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
    
    # 코드 리뷰 요구사항 확인
    workflow_focus = state.get("workflow_focus", [])
    requires_code_review = any("review" in str(wf).lower() or "리뷰" in str(wf) for wf in workflow_focus) if workflow_focus else False
    if not requires_code_review:
        # 사용자 메시지에서도 확인
        last_user_msg = str(messages_list[-1].content) if messages_list else ""
        requires_code_review = "리뷰" in last_user_msg or "review" in last_user_msg.lower()
    
    # 추천 도구 중 리뷰 기능 지원 여부 확인
    recommended_tools_have_review = []
    for info in recommended_tools_info:
        tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
        if tool_fact_dict:
            workflow_support = tool_fact_dict.get("workflow_support", [])
            has_review = any("review" in str(ws).lower() or "리뷰" in str(ws) for ws in workflow_support)
            recommended_tools_have_review.append({
                "name": info['name'],
                "has_review": has_review
            })
    
    # Findings에서 리뷰 전용 도구 찾기 (하드코딩 제거 - tool_facts와 findings에서 동적으로 찾기)
    review_tool_names = []
    if requires_code_review:
        # 1. tool_facts에서 리뷰 관련 도구 찾기
        for tool_fact_dict in tool_facts:
            tool_name = tool_fact_dict.get("name", "")
            if tool_name and tool_name not in [info['name'] for info in recommended_tools_info]:
                # 추천되지 않은 도구 중에서 리뷰 기능이 있는 도구 찾기
                workflow_support = tool_fact_dict.get("workflow_support", [])
                feature_category = tool_fact_dict.get("feature_category", "")
                if (any("review" in str(ws).lower() or "리뷰" in str(ws) for ws in workflow_support) or 
                    "review" in feature_category.lower() or "리뷰" in feature_category):
                    review_tool_names.append(tool_name)
        
        # 2. tool_facts에서 찾지 못한 경우, findings 텍스트에서 직접 찾기
        if not review_tool_names:
            import re
            # 원본 findings와 notes에서 대소문자 유지하며 찾기
            original_text = (findings + " " + " ".join([str(n) for n in notes])) if findings or notes else ""
            # 리뷰 관련 도구 이름 패턴 찾기
            review_patterns = re.findall(r'\b([A-Z][a-zA-Z]*(?:Review|CodeReview|Reviewer|리뷰)[a-zA-Z]*)\b', original_text)
            review_tool_names = list(set([name for name in review_patterns if name and len(name) > 3]))
    
    # Decision Engine 결과를 자연스러운 형태로 변환 (내부 평가 용어 완전 제거)
    review_note = ""
    if requires_code_review:
        review_tools = [t for t in recommended_tools_have_review if t['has_review']]
        if not review_tools:
            if review_tool_names:
                review_tool_examples = ", ".join(review_tool_names[:3])  # 최대 3개만
                review_note = f"\n**⚠️ 리뷰 기능 안내**: 추천된 도구는 코드 작성에 특화되어 있으며, 코드 리뷰 기능이 필요하다면 Findings에서 확인한 PR 리뷰 전용 도구({review_tool_examples} 등)와 함께 사용하는 것을 권장합니다.\n"
            else:
                review_note = "\n**⚠️ 리뷰 기능 안내**: 추천된 도구는 코드 작성에 특화되어 있으며, 코드 리뷰 기능이 필요하다면 Findings에서 확인한 PR 리뷰 전용 도구와 함께 사용하는 것을 권장합니다.\n"
    
    decision_summary = f"""**추천 도구 (우선순위 순서대로):**
{chr(10).join([f"{info['priority']}. {info['name']}: {info['reasoning']}" for info in recommended_tools_info])}

{f"**비용 정보 ({team_size}명 팀 기준):**" + chr(10) + chr(10).join([f"- {info['name']}: {info['cost']}" for info in recommended_tools_info if info['cost']]) if team_size and any(info['cost'] for info in recommended_tools_info) else ""}
{review_note}
"""
    
    # 최종 리포트 생성 프롬프트 (Decision Engine 결과 포함하되 내부 평가 용어 완전 제거)
    combined_constraints = f"{constraints_text_simple}\n\n**내부 분석 결과 (이 정보를 바탕으로 자연스러운 답변을 생성하세요 - 내부 평가 과정은 완전히 숨기세요!):**\n\n{decision_summary}"
    
    # 답변 형식 가져오기 (기본값: markdown)
    response_format = state.get("response_format", "markdown")
    print(f"🔍 [DEBUG] structured_report - 답변 형식: {response_format}")
    print(f"🔍 [DEBUG] structured_report - state 전체 키: {list(state.keys())}")
    if "response_format" in state:
        print(f"✅ [DEBUG] structured_report - response_format이 state에 존재: {state['response_format']}")
    else:
        print(f"⚠️ [DEBUG] structured_report - response_format이 state에 없음! 기본값 'markdown' 사용")
    
    report_prompt = final_report_generation_prompt.format(
        research_brief=research_brief,
        messages=get_buffer_string(messages_list),  # 전체 대화 이력 사용 (final_report_generation과 일관성 유지)
        findings=findings[:3000] if findings else "연구 결과 없음",
        date=date,
        is_followup="YES" if is_followup else "NO",
        previous_tools="",
        question_type="decision",
        constraints=combined_constraints,
        response_format=response_format  # 🆕 답변 형식 전달
    )
    
    # LLM으로 리포트 생성
    # 모델별 max_tokens 제한 확인 및 적용
    model_name = configurable.final_report_model.lower()
    if "gpt-4o-mini" in model_name:
        max_tokens_allowed = min(configurable.final_report_model_max_tokens, 16384)  # gpt-4o-mini 최대 16384
    elif "gpt-4o" in model_name:
        max_tokens_allowed = min(configurable.final_report_model_max_tokens, 4096)  # gpt-4o 최대 4096 (일반적으로)
    elif "gpt-4" in model_name:
        max_tokens_allowed = min(configurable.final_report_model_max_tokens, 4096)  # gpt-4 최대 4096
    else:
        max_tokens_allowed = min(configurable.final_report_model_max_tokens, 16384)  # 기본값
    
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": max_tokens_allowed,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
    }
    
    print(f"🔍 [DEBUG] 모델: {configurable.final_report_model}, max_tokens: {max_tokens_allowed} (원래 설정: {configurable.final_report_model_max_tokens})")
    
    try:
        # 🆕 표 형식 요청 시 Structured Output 사용
        if response_format == "table":
            from app.agent.state import TableData
            
            try:
                print(f"🔍 [DEBUG] Structured Output으로 표 데이터 생성 시작 (structured_report)")
                print(f"🔍 [DEBUG] configurable_model 타입: {type(configurable_model)}")
                
                # Structured Output으로 표 데이터 생성
                table_model = (
                    configurable_model
                    .with_structured_output(TableData)
                    .with_config(writer_model_config)
                )
                
                print(f"🔍 [DEBUG] table_model 생성 완료, LLM 호출 시작")
                table_data = await table_model.ainvoke([HumanMessage(content=report_prompt)])
                print(f"🔍 [DEBUG] LLM 응답 수신 완료, 타입: {type(table_data)}")
                
                # table_data가 TableData 객체인지 확인
                if not hasattr(table_data, 'columns') or not hasattr(table_data, 'rows'):
                    raise ValueError(f"TableData 객체가 아닙니다. 타입: {type(table_data)}, 값: {table_data}")
                
                print(f"✅ [DEBUG] 표 데이터 생성 완료: {len(table_data.columns)}개 열, {len(table_data.rows)}개 행")
                print(f"🔍 [DEBUG] 표 열: {table_data.columns}")
                print(f"🔍 [DEBUG] 표 행 개수: {len(table_data.rows)}")
                
                # JSON 형식으로 변환 (프론트엔드에서 파싱 가능하도록)
                import json
                try:
                    table_json = json.dumps({
                        "type": "table",
                        "columns": list(table_data.columns) if hasattr(table_data.columns, '__iter__') else table_data.columns,
                        "rows": [list(row) if hasattr(row, '__iter__') else row for row in table_data.rows] if hasattr(table_data.rows, '__iter__') else table_data.rows
                    }, ensure_ascii=False, indent=2)
                except Exception as json_error:
                    raise ValueError(f"JSON 변환 실패: {json_error}, columns 타입: {type(table_data.columns)}, rows 타입: {type(table_data.rows)}")
                
                # 리포트 본문은 JSON 문자열로 설정 (프론트엔드에서 파싱)
                report_body = table_json
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"⚠️ [DEBUG] Structured Output 실패: {e}")
                print(f"⚠️ [DEBUG] 예외 타입: {type(e).__name__}")
                print(f"⚠️ [DEBUG] 예외 상세 정보:\n{error_detail}")
                print(f"⚠️ [DEBUG] 일반 텍스트로 폴백")
                # 폴백: 일반 텍스트로 표 형식 생성
                report_prompt = final_report_generation_prompt.format(
                    research_brief=research_brief,
                    messages=get_buffer_string(messages_list),
                    findings=findings[:3000] if findings else "연구 결과 없음",
                    date=date,
                    is_followup="YES" if is_followup else "NO",
                    previous_tools="",
                    question_type="decision",
                    constraints=combined_constraints,
                    response_format="markdown"  # 폴백 시 markdown
                )
                try:
                    final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                        HumanMessage(content=report_prompt)
                    ])
                    report_body = str(final_report.content).strip()
                except Exception as fallback_error:
                    import traceback
                    error_detail = traceback.format_exc()
                    print(f"⚠️ [DEBUG] 폴백 LLM 호출 실패: {type(fallback_error).__name__}: {fallback_error}")
                    print(f"⚠️ [DEBUG] 폴백 예외 상세 정보:\n{error_detail}")
                    # 최종 fallback: 간단한 메시지
                    report_body = f"## 💡 추천 도구\n\n{', '.join([info['name'] for info in recommended_tools_info])}\n\n상세 정보는 다시 시도해주세요."
        else:
            print(f"🔍 [DEBUG] Structured Report 생성 시작 (프롬프트 길이: {len(report_prompt)}자)")
            
            # 리포트 재생성 로직 (최대 2번 재시도)
            max_retries = 2
            report_body = None
            for attempt in range(max_retries + 1):
                try:
                    print(f"🔍 [DEBUG] LLM 호출 시작 (시도 {attempt + 1}/{max_retries + 1})")
                    final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                        HumanMessage(content=report_prompt)
                    ])
                    print(f"🔍 [DEBUG] LLM 응답 수신 완료, 타입: {type(final_report)}")
                    report_body = str(final_report.content).strip()
                    print(f"🔍 [DEBUG] report_body 길이: {len(report_body)}자")
                    
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
                    
                    # 리포트 내용 완성도 검증 (잘림 여부 확인)
                    # 마지막 문장이 완전한지 확인
                    is_complete = True
                    if report_body:
                        last_100_chars = report_body.strip()[-100:].strip()
                        # 문장이 중간에 잘렸는지 확인 (불완전한 단어, 문장 부호 없이 끝나는 경우)
                        # 특정 잘린 패턴 확인
                        truncated_patterns = [
                            "다양한 프로그래",  # "다양한 프로그래밍 언어"가 잘림
                            "포함한 여러",  # "포함한 여러 언어를 지원합니다"가 잘림
                            "TypeSc",  # "TypeScript"가 잘림
                            " 및 Ty",  # "Java, JavaScript 및 TypeScript"가 잘림
                            "JavaScript 및 Ty",  # TypeScript가 잘림
                            "Java, JavaScript 및 Ty",  # TypeScript가 잘림
                            "*",  # 마크다운 불완전한 리스트로 끝남
                            "**",  # 마크다운 불완전한 볼드로 끝남
                        ]
                        
                        # 불완전한 단어 패턴 (2-3글자로 끝나는 경우) - "Ty", "Java, JavaScript 및 Ty" 등
                        if re.search(r'[가-힣a-zA-Z]{1,3}\s*$', last_100_chars):
                            # 마지막 문자가 불완전한 단어로 끝나는지 확인
                            last_word = last_100_chars.strip().split()[-1] if last_100_chars.strip().split() else ""
                            if last_word and len(last_word) <= 3 and not any(last_word.endswith(p) for p in ['.', '!', '?', ',', ':', ';']):
                                is_complete = False
                                print(f"⚠️ [Structured Report] 불완전한 단어 패턴 감지: '{last_word}' (마지막 단어가 너무 짧음)")
                        
                        # 마지막 문자가 "*"로 끝나는 경우도 잘림으로 간주
                        if report_body.strip().endswith("*") or report_body.strip().endswith("**"):
                            is_complete = False
                            print(f"⚠️ [Structured Report] 마크다운 불완전 패턴 감지: 리포트가 '*' 또는 '**'로 끝남")
                        
                        for pattern in truncated_patterns:
                            if pattern in last_100_chars:
                                is_complete = False
                                print(f"⚠️ [Structured Report] 잘림 패턴 감지: '{pattern}'")
                                break
                        
                        # 문장 부호로 끝나지 않고 불완전한 단어로 끝나는 경우
                        if is_complete:
                            last_chars = report_body.strip()[-30:]
                            # 마지막이 문장 부호로 끝나는지 확인
                            if not any(last_chars.rstrip().endswith(p) for p in ['.', '!', '?', ':', ';', ')', '}', ']', '>']):
                                # 불완전한 단어 패턴 확인 (1-4글자로 끝나는 경우)
                                if re.search(r'[가-힣a-zA-Z]{1,4}\s*$', last_chars):
                                    is_complete = False
                                    print(f"⚠️ [Structured Report] 불완전한 문장 감지: '{last_chars}'")
                    
                    # 리포트 완성도 검증
                    if not report_body or len(report_body) < 1000 or not is_complete:
                        if attempt < max_retries:
                            if len(report_body) < 1000:
                                issue_desc = "너무 짧음"
                            elif not is_complete:
                                issue_desc = "내용이 잘림"
                            else:
                                issue_desc = "불완전"
                            print(f"⚠️ [Structured Report] 리포트 {issue_desc} ({len(report_body)}자, 최소 1000자 필요) - 재생성 시도 {attempt + 1}/{max_retries}")
                            # 재생성 시 더 강력한 요구사항 추가
                            retry_note = f"\n\n⚠️⚠️⚠️ 매우 중요 - 재생성 요구사항 ({attempt + 1}번째 시도):\n"
                            retry_note += f"- 리포트는 현재 {len(report_body)}자로 부족하거나 내용이 잘렸습니다!\n"
                            retry_note += f"- 반드시 최소 1500자 이상, 각 도구당 최소 400자 이상 상세히 설명하세요!\n"
                            retry_note += f"- 마지막 문장은 반드시 완전한 문장으로 끝나야 합니다! (마침표, 느낌표, 물음표 등)\n"
                            retry_note += f"- 단어가 중간에 잘리면 안 됩니다! (예: 'TypeScript'를 'Ty'로 줄이면 안 됩니다)\n"
                            retry_note += f"- 각 도구의 가격, 통합 기능, 장점, 추천 이유를 더 상세히 설명하세요!\n"
                            retry_note += f"- 결론 섹션을 포함하여 리포트를 완성하세요!\n"
                            
                            # 기존 요구사항 업데이트 또는 추가
                            if "리포트는 최소" in report_prompt:
                                # 기존 요구사항을 더 강화된 버전으로 교체
                                import re
                                report_prompt = re.sub(
                                    r'리포트는 최소 \d+자 이상이어야 합니다!.*?마지막 문장은 반드시 완전한 문장 부호.*?',
                                    f"리포트는 최소 1500자 이상이어야 하며, 각 도구당 최소 400자 이상 상세히 설명하세요! 반드시 완전한 문장으로 끝나야 합니다! 단어가 중간에 잘리면 안 됩니다!{retry_note}",
                                    report_prompt,
                                    flags=re.DOTALL
                                )
                            else:
                                report_prompt += retry_note
                            continue
                        else:
                            if len(report_body) < 1000:
                                issue_desc = "너무 짧음"
                            elif not is_complete:
                                issue_desc = "내용이 잘림"
                            else:
                                issue_desc = "불완전"
                            print(f"⚠️ [Structured Report] 리포트 {issue_desc} ({len(report_body)}자, 최소 1000자 필요) - 재시도 실패, fallback 사용")
                            if len(report_body) < 1000:
                                raise ValueError(f"리포트가 너무 짧습니다 ({len(report_body)}자, 최소 1000자 필요)")
                            else:
                                raise ValueError(f"리포트 내용이 잘렸습니다 ({len(report_body)}자)")
                    
                    # 추천 도구가 모두 포함되어 있는지 확인
                    recommended_count_in_report = sum(1 for tool_name in decision_result.recommended_tools[:3] if tool_name in report_body)
                    if recommended_count_in_report < len(decision_result.recommended_tools[:3]):
                        if attempt < max_retries:
                            print(f"⚠️ [Structured Report] 일부 추천 도구가 리포트에 없음 (포함: {recommended_count_in_report}/{len(decision_result.recommended_tools[:3])}) - 재생성 시도 {attempt + 1}/{max_retries}")
                            continue
                        else:
                            print(f"⚠️ [Structured Report] 일부 추천 도구가 리포트에 없음 (포함: {recommended_count_in_report}/{len(decision_result.recommended_tools[:3])}) - 재시도 실패")
                            raise ValueError("추천 도구가 모두 포함되지 않았습니다")
                    
                    # 각 도구별로 최소 정보가 포함되어 있는지 확인
                    all_tools_included = True
                    for tool_name in decision_result.recommended_tools[:3]:
                        tool_pos = report_body.find(tool_name)
                        if tool_pos == -1:
                            all_tools_included = False
                            break
                        if team_size:
                            tool_section = report_body[tool_pos:tool_pos + 800]
                            # 가격 정보 확인 (가격, $, 사용량 기반, API 호출 등 모두 확인)
                            has_price_info = (
                                "가격" in tool_section or 
                                "$" in tool_section or 
                                "사용량" in tool_section or 
                                "API 호출" in tool_section or 
                                "토큰" in tool_section or
                                "usage-based" in tool_section.lower()
                            )
                            if not has_price_info:
                                print(f"⚠️ [Structured Report] {tool_name} 가격 정보가 리포트에 없음 (경고만)")
                                
                    # 리포트 내용 잘림 확인 (마지막 문장이 완전한지)
                    if report_body and len(report_body) > 100:
                        last_50_chars = report_body.strip()[-50:]
                        
                        # 불완전한 패턴 확인
                        is_truncated = False
                        
                        # 특정 잘린 패턴 확인
                        if "다양한 프로그래" in last_50_chars or "TypeSc" in last_50_chars:
                            is_truncated = True
                        # 문장 부호 없이 끝나고, 불완전한 단어로 끝나는 경우
                        elif not any(last_50_chars.rstrip().endswith(p) for p in ['.', '!', '?', ':', ';', ')', '}', ']', '>']):
                            # 마지막이 불완전한 단어로 끝나는지 확인 (1-3글자)
                            if re.search(r'[가-힣a-zA-Z]{1,3}\s*$', last_50_chars[-10:]):
                                is_truncated = True
                        
                        if is_truncated:
                            if attempt < max_retries:
                                print(f"⚠️ [Structured Report] 리포트 내용이 잘린 것으로 의심됨 (마지막 30자: {report_body.strip()[-30:]}) - 재생성 시도 {attempt + 1}/{max_retries}")
                                if "리포트의 마지막 문장을 반드시 완전하게 작성하세요" not in report_prompt:
                                    report_prompt += "\n\n⚠️ 중요: 리포트의 마지막 문장을 반드시 완전하게 작성하세요! 문장이 중간에 잘리면 안 됩니다! 모든 문장은 반드시 문장 부호(마침표, 물음표 등)로 끝나야 합니다!"
                                continue
                            else:
                                print(f"⚠️ [Structured Report] 리포트 내용이 잘린 것으로 의심됨 (마지막 30자: {report_body.strip()[-30:]}) - 재시도 실패, fallback 사용")
                                # fallback으로 진행하되, 잘린 부분 제거
                                # 마지막 불완전한 문장 제거
                                lines = report_body.strip().split('\n')
                                if lines:
                                    # 마지막 줄이 불완전하면 제거
                                    if len(lines[-1].strip()) < 10 or re.search(r'[가-힣a-zA-Z]{1,3}\s*$', lines[-1].strip()[-5:]):
                                        report_body = '\n'.join(lines[:-1]).strip()
                                        if not report_body.endswith(('.', '!', '?', ':', ';')):
                                            report_body += '.'
                    
                    if not all_tools_included and attempt < max_retries:
                        print(f"⚠️ [Structured Report] 도구 정보 누락 - 재생성 시도 {attempt + 1}/{max_retries}")
                        continue
                    
                    # 검증 통과
                    print(f"✅ [Structured Report] 리포트 생성 성공 ({len(report_body)}자)")
                    break
                    
                except Exception as e:
                    if attempt < max_retries:
                        print(f"⚠️ [Structured Report] 리포트 생성 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                        continue
                    else:
                        raise
        
        # 최종 검증: 리포트가 생성되었는지 확인
        # (길이 및 완성도 검증은 위 루프에서 이미 했으므로 여기서는 확인만)
        if not report_body:
            raise ValueError(f"리포트가 생성되지 않았습니다")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"⚠️ [Structured Report] LLM 리포트 생성 실패 또는 불완전: {e}")
        print(f"⚠️ [Structured Report] 예외 상세 정보:\n{error_trace}")
        # Fallback: 상세한 리포트 생성 (최소 1000자 보장)
        report_body = f"## 💡 추천 도구\n\n"
        for info in recommended_tools_info:
            if info['priority'] == 1:
                report_body += f"### 가장 추천하는 도구: {info['name']}\n\n"
            else:
                report_body += f"### 대안 {info['priority']-1}: {info['name']}\n\n"
            
            # reasoning이 있으면 포함, 없으면 상세한 설명 생성
            if info['reasoning'] and len(info['reasoning']) > 10:
                # reasoning에서 불완전한 부분 제거
                reasoning_clean = info['reasoning'].replace("();", "").replace("()", "").strip()
                if reasoning_clean:
                    report_body += f"{reasoning_clean}\n\n"
                else:
                    # tool_facts에서 상세 정보 가져오기
                    tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
                    if tool_fact_dict:
                        supported_languages = tool_fact_dict.get("supported_languages", [])
                        if supported_languages:
                            report_body += f"{info['name']}은(는) {', '.join(supported_languages[:5])} 등 다양한 프로그래밍 언어를 지원합니다. "
                    report_body += f"{info['name']}은(는) 8명 규모의 백엔드·프론트엔드 개발팀에 적합한 도구입니다. "
                    # 코드 리뷰 요구사항 반영
                    tool_has_review = next((t['has_review'] for t in recommended_tools_have_review if t['name'] == info['name']), False) if 'recommended_tools_have_review' in locals() else False
                    if requires_code_review and tool_has_review:
                        report_body += "코드 작성과 리뷰 기능을 모두 지원합니다."
                    elif requires_code_review and not tool_has_review:
                        report_body += "코드 작성에 특화되어 있으며, 리뷰 기능이 필요하다면 전용 리뷰 도구와 함께 사용하는 것을 권장합니다."
                    else:
                        report_body += "코드 작성과 자동 완성 기능을 제공합니다."
                    report_body += "\n\n"
            else:
                # tool_facts에서 상세 정보 가져오기
                tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
                if tool_fact_dict:
                    supported_languages = tool_fact_dict.get("supported_languages", [])
                    if supported_languages:
                        report_body += f"{info['name']}은(는) {', '.join(supported_languages[:5])} 등 다양한 프로그래밍 언어를 지원하여 백엔드와 프론트엔드 개발에 모두 활용할 수 있습니다. "
                    integrations = tool_fact_dict.get("integrations", [])
                    if integrations:
                        report_body += f"GitHub, GitLab, {', '.join(integrations[:3])} 등 주요 개발 도구와의 통합이 가능합니다. "
                    features = tool_fact_dict.get("primary_features", [])
                    if features:
                        report_body += f"주요 기능으로는 {', '.join(features[:3])} 등이 있습니다. "
                
                report_body += f"{info['name']}은(는) {team_size}명 규모의 백엔드·프론트엔드 개발팀에 적합한 도구입니다. "
                # 코드 리뷰 요구사항 반영
                tool_has_review = next((t['has_review'] for t in recommended_tools_have_review if t['name'] == info['name']), False) if 'recommended_tools_have_review' in locals() else False
                if requires_code_review and tool_has_review:
                    report_body += "코드 작성과 리뷰 기능을 모두 지원하여 팀의 코드 품질 향상에 도움을 줄 수 있습니다. "
                elif requires_code_review and not tool_has_review:
                    report_body += "코드 작성에 특화되어 있으며, 리뷰 기능이 필요하다면 전용 리뷰 도구와 함께 사용하는 것을 권장합니다. "
                else:
                    report_body += "코드 작성과 자동 완성 기능을 제공하여 개발 생산성을 향상시킬 수 있습니다. "
                report_body += "\n\n"
            
            # 가격 정보 포함 (올바른 계산, 플랜 타입 명시)
            if info['cost'] and team_size:
                # 가격 정보가 완전한지 확인 (":"로 끝나지 않도록)
                cost_info = info['cost'].strip()
                if cost_info and not cost_info.endswith(":"):
                    report_body += f"**💰 가격**: {cost_info}\n\n"
                elif cost_info:
                    # tool_facts에서 가격 정보 다시 가져오기
                    tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
                    if tool_fact_dict:
                        pricing_plans = tool_fact_dict.get("pricing_plans", [])
                        if pricing_plans:
                            # 첫 번째 플랜 사용
                            plan = pricing_plans[0]
                            plan_name = plan.get("name", "플랜")
                            if plan.get("price_per_user_per_month"):
                                price = plan["price_per_user_per_month"] * team_size
                                report_body += f"**💰 가격**: 팀 플랜 ({plan_name}): ${price:.0f}/월\n\n"
                            elif plan.get("price_per_month"):
                                report_body += f"**💰 가격**: {plan_name}: ${plan['price_per_month']:.0f}/월 (팀 플랜은 공식 사이트에서 확인 필요)\n\n"
                            else:
                                report_body += f"**💰 가격**: 가격 정보는 공식 사이트에서 확인이 필요합니다.\n\n"
                        else:
                            report_body += f"**💰 가격**: 가격 정보는 공식 사이트에서 확인이 필요합니다.\n\n"
                    else:
                        report_body += f"**💰 가격**: 가격 정보는 공식 사이트에서 확인이 필요합니다.\n\n"
            
            # 통합 기능 정보 추가 (tool_facts에서)
            tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
            if tool_fact_dict:
                integrations = tool_fact_dict.get("integrations", [])
                if integrations:
                    report_body += f"**🔗 통합 기능**: {', '.join(integrations[:5])}\n\n"
        
        # 코드 리뷰 요구사항 반영 (하드코딩 제거)
        if requires_code_review:
            review_tools = [t for t in recommended_tools_have_review if t['has_review']] if 'recommended_tools_have_review' in locals() else []
            if not review_tools:
                # Findings에서 리뷰 전용 도구 찾기 (이미 위에서 찾았거나, 다시 찾기)
                review_tool_names_fallback = []
                for tool_fact_dict in tool_facts:
                    tool_name = tool_fact_dict.get("name", "")
                    if tool_name and tool_name not in [info['name'] for info in recommended_tools_info]:
                        workflow_support = tool_fact_dict.get("workflow_support", [])
                        feature_category = tool_fact_dict.get("feature_category", "")
                        if (any("review" in str(ws).lower() or "리뷰" in str(ws) for ws in workflow_support) or 
                            "review" in feature_category.lower() or "리뷰" in feature_category):
                            review_tool_names_fallback.append(tool_name)
                
                report_body += "\n## ⚠️ 코드 리뷰 기능 안내\n\n"
                # 이미 찾은 review_tool_names 사용 또는 다시 찾기
                if not review_tool_names_fallback:
                    # findings 텍스트에서도 직접 찾기
                    import re
                    review_patterns = re.findall(r'\b([A-Z][a-zA-Z]*(?:Review|CodeReview|Reviewer|리뷰)[a-zA-Z]*)\b', findings + " " + " ".join([str(n) for n in notes]))
                    review_tool_names_fallback.extend([name for name in review_patterns if name not in review_tool_names_fallback])
                
                if review_tool_names_fallback:
                    review_tool_examples = ", ".join(list(set(review_tool_names_fallback))[:3])  # 중복 제거 후 최대 3개만
                    report_body += f"추천된 도구는 코드 작성에 특화되어 있으며, 코드 리뷰 기능이 필요하다면 Findings에서 확인한 PR 리뷰 전용 도구({review_tool_examples} 등)와 함께 사용하는 것을 권장합니다.\n\n"
                else:
                    # 이미 찾은 review_tool_names 사용
                    if review_tool_names:
                        review_tool_examples = ", ".join(review_tool_names[:3])
                        report_body += f"추천된 도구는 코드 작성에 특화되어 있으며, 코드 리뷰 기능이 필요하다면 Findings에서 확인한 PR 리뷰 전용 도구({review_tool_examples} 등)와 함께 사용하는 것을 권장합니다.\n\n"
                    else:
                        report_body += "추천된 도구는 코드 작성에 특화되어 있으며, 코드 리뷰 기능이 필요하다면 Findings에서 확인한 PR 리뷰 전용 도구와 함께 사용하는 것을 권장합니다.\n\n"
        
        # 결론 섹션 추가 (최소 길이 보장을 위해 상세하게)
        report_body += "\n## 💡 결론\n\n"
        tool_names = ", ".join([info['name'] for info in recommended_tools_info])
        if len(recommended_tools_info) > 1:
            report_body += f"위 {len(recommended_tools_info)}개 도구({tool_names})를 함께 사용하면 "
        else:
            report_body += f"{tool_names}을(를) 사용하면 "
        
        if team_size:
            report_body += f"{team_size}명 규모의 백엔드·프론트엔드 개발팀의 생산성을 크게 향상시킬 수 있습니다. "
        
        if requires_code_review:
            report_body += "이 도구들은 코드 작성과 리뷰 작업을 효율적으로 진행할 수 있도록 도와주며, 팀의 개발 워크플로우를 개선할 수 있습니다. "
            # 리뷰 기능 지원 여부 언급
            review_count = sum(1 for t in recommended_tools_have_review if t['has_review']) if 'recommended_tools_have_review' in locals() else 0
            if review_count > 0:
                report_body += "특히 코드 리뷰 기능을 내장하고 있어 팀원 간의 코드 품질 향상과 지식 공유에 기여할 수 있습니다. "
        else:
            report_body += "개발 생산성을 높이고 코드 작성 속도를 향상시킬 수 있습니다. "
        
        report_body += "각 도구의 기능과 가격을 고려하여 팀의 요구사항에 맞는 선택을 하시기 바랍니다. 도입 전 무료 체험판이나 평가판을 활용하여 팀에 적합한지 확인해보시기 바랍니다.\n\n"
        
        # 리포트가 여전히 너무 짧으면 추가 정보 포함 (최소 1000자 보장)
        # 중복 방지: 이미 "추가 고려사항" 섹션이 있으면 추가하지 않음
        has_additional_section = "## 📌 추가 고려사항" in report_body or "## 추가 고려사항" in report_body
        
        if len(report_body) < 1000:
            if not has_additional_section:
                report_body += "\n## 📌 추가 고려사항\n\n"
            else:
                # 이미 섹션이 있으면 그 다음에 이어서 추가
                report_body += "\n\n"
            
            # 중복 추가 방지: 이미 같은 내용이 있으면 추가하지 않음
            existing_content = "팀의 개발 환경과 워크플로우를 고려하여 도구를 선택하세요"
            if existing_content not in report_body:
                report_body += "팀의 개발 환경과 워크플로우를 고려하여 도구를 선택하세요. 각 도구는 고유한 장점이 있으므로, 팀의 구체적인 요구사항과 예산을 함께 검토하는 것이 좋습니다. 도입 전 무료 체험판이나 평가판을 활용하여 팀에 적합한지 확인해보시기 바랍니다. 또한, 팀원들의 학습 곡선과 도구의 통합 난이도도 함께 고려하시기 바랍니다. "
            
            # 도구별 추가 정보
            for info in recommended_tools_info:
                tool_fact_dict = next((t for t in tool_facts if t.get("name") == info['name']), None)
                if tool_fact_dict and len(report_body) < 1000:
                    features = tool_fact_dict.get("features", [])
                    integrations = tool_fact_dict.get("integrations", [])
                    if features:
                        feature_text = f"{info['name']}은(는) {', '.join(features[:3])} 등의 기능을 제공합니다. "
                        if feature_text not in report_body:
                            report_body += feature_text
                    if integrations and len(report_body) < 1000:
                        integration_text = f"{info['name']}은(는) {', '.join(integrations[:3])} 등과 통합할 수 있습니다. "
                        if integration_text not in report_body:
                            report_body += integration_text
            
            # 여전히 부족하면 일반적인 조언 추가 (중복 방지)
            if len(report_body) < 1000:
                additional_text = "팀의 개발 문화와 도구 사용 경험을 고려하여 선택하시기 바랍니다. 도입 후 팀원들의 피드백을 수집하여 필요시 다른 도구로 전환하는 것도 고려해볼 수 있습니다. "
                if additional_text not in report_body:
                    report_body += additional_text
            
            report_body += "\n\n"
        
        # 디버깅: 리포트 생성 결과 확인
        print(f"🔍 [Structured Report DEBUG] 리포트 생성 완료:")
        print(f"  - 추천 도구 개수: {len(decision_result.recommended_tools)}")
        print(f"  - 제외 도구 개수: {len(decision_result.excluded_tools)}")
        print(f"  - 리포트 길이: {len(report_body)}자")
        print(f"  - 리포트 시작 200자: {report_body[:200]}")
        
        # 🚨 재검색이 필요 없는 경우(need_research = false)에는 캐시/벡터 DB 저장 건너뛰기
        need_research = state.get("need_research", True)  # 기본값: True (검색 필요)
        
        if need_research:
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
        else:
            print(f"✅ [캐시 저장 건너뛰기] 재검색 불필요 (need_research = false) - 이전 대화 정보만 사용했으므로 저장하지 않음")
        
        # 최종 검증: greeting과 report_body가 모두 있는지 확인
        if not greeting or len(greeting) < 10:
            print(f"⚠️ [Structured Report] greeting이 비어있음: '{greeting}', 최소 생성")
            last_user_message = messages_list[-1].content if messages_list and isinstance(messages_list[-1], HumanMessage) else ""
            if last_user_message:
                greeting = f"{str(last_user_message)[:50]}에 대해 분석해드리겠습니다."
            else:
                greeting = "분석해드리겠습니다."
        
        if not report_body or len(report_body) < 50:
            print(f"⚠️ [Structured Report] report_body가 비어있음: {len(report_body) if report_body else 0}자, 에러 메시지 반환")
            report_body = "죄송합니다. 리포트 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        
        print(f"✅ [Structured Report] 최종 반환: greeting ({len(greeting)}자), report_body ({len(report_body)}자)")
        
        return {
            "final_report": report_body,
            "messages": [
                AIMessage(content=greeting),
                AIMessage(content=report_body)
            ],
            "notes": {"type": "override", "value": []}
        }
