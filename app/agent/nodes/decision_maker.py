"""Decision Engine 실행 노드 - run_decision_engine"""

import re
from datetime import datetime

from app.agent.nodes._common import (
    RunnableConfig,
    AgentState,
    Configuration,
    HumanMessage,
    extract_tool_facts,
    DecisionEngine,
    ToolFact,
    UserContext,
    WorkflowType,
)


async def run_decision_engine(state: AgentState, config: RunnableConfig):
    """Decision Engine 실행 (의사결정 질문인 경우)"""
    
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
    
    # 🚨 전체 사용자 메시지 히스토리에서 정보 추출 (HumanMessage만)
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    all_user_messages_text = " ".join([str(msg.content).lower() for msg in human_messages])
    
    # 팀 규모 추출 시도 (전체 히스토리에서)
    if not team_size and all_user_messages_text:
        # "개인", "개인 개발자", "개인 사용자" 등을 인식하여 team_size = 1로 설정
        if any(keyword in all_user_messages_text for keyword in ["개인", "개인 개발자", "개인 사용자", "개인용", "개인으로"]):
            team_size = 1
        else:
            # "X명" 패턴 찾기
            team_size_match = re.search(r'(\d+)\s*명', all_user_messages_text)
            if team_size_match:
                team_size = int(team_size_match.group(1))
    
    # 예산 추출 시도 (전체 히스토리에서)
    if not budget_max and all_user_messages_text:
        budget_patterns = [
            r'월\s*\$?\s*(\d+)',
            r'\$?\s*(\d+)\s*까지',
            r'\$?\s*(\d+)\s*가능',
            r'\$?\s*(\d+)\s*이하',
            r'\$?\s*(\d+)\s*이내',
        ]
        for pattern in budget_patterns:
            budget_match = re.search(pattern, all_user_messages_text)
            if budget_match:
                budget_max = float(budget_match.group(1))
                break
    
    # 개발 언어/분야 확인 (제약 조건이 없어도 개발 언어/분야가 있으면 충분!)
    has_development_area = False
    if all_user_messages_text:
        # 프로그래밍 언어 확인 (하드코딩 - 언어는 정해져 있으므로 OK)
        languages = ["python", "javascript", "java", "typescript", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "r", "scala", "clojure", "perl", "lua", "matlab"]
        # 개발 분야 확인
        domains = ["웹 개발", "백엔드", "프론트엔드", "풀스택", "모바일", "게임", "데이터", "ai", "ml", "머신러닝", "앱 개발"]
        # 프레임워크/라이브러리 확인
        frameworks = ["react", "vue", "angular", "django", "flask", "spring", "node.js", "express", "fastapi", "laravel", "rails"]
        
        if any(lang in all_user_messages_text for lang in languages) or \
           any(domain in all_user_messages_text for domain in domains) or \
           any(fw in all_user_messages_text for fw in frameworks) or \
           re.search(r'으로\s*개발|로\s*개발|개발', all_user_messages_text):
            has_development_area = True
    
    # 🚨 기본적으로 정보가 충분하다고 가정!
    # 정말 모호한 경우만 명확화 요구
    # 다음 중 하나라도 있으면 충분한 정보:
    # 1. 개발 언어/분야가 있음
    # 2. 사용 형태(개인/팀)가 있음
    # 3. 제약 조건(예산/팀 규모)이 있음
    # 4. 일반적인 추천 요청 (코딩 AI 도구 추천 등)
    
    # 정말 모호한 경우 체크 (명확화 필요)
    is_too_vague = False
    if all_user_messages_text:
        # 너무 모호한 표현들
        vague_patterns = [
            r'나\s*개발\s*할건데',  # "나 개발 할건데"
            r'개발\s*할건데',  # "개발 할건데"
            r'개발\s*하려고\s*하는데',  # "개발 하려고 하는데"
            r'개발\s*하려는데',  # "개발 하려는데"
        ]
        # 모호한 패턴이 있고, 다른 구체적인 정보가 없으면 모호함
        has_vague_pattern = any(re.search(pattern, all_user_messages_text) for pattern in vague_patterns)
        has_user_type_keyword = any(keyword in all_user_messages_text for keyword in ["개인", "개인 개발자", "개인 사용자", "개인용", "개인으로", "팀", "팀용", "우리 팀"])
        if has_vague_pattern and not has_development_area and not has_user_type_keyword and not team_size and not budget_max:
            is_too_vague = True
    
    # 정보 충분 여부 판단: 모호하지 않고, 어느 정도 정보가 있으면 충분
    has_sufficient_info = not is_too_vague and (
        has_development_area or  # 개발 언어/분야가 있으면 충분
        team_size is not None or  # 팀 규모가 있으면 충분
        budget_max is not None or  # 예산이 있으면 충분
        any(keyword in all_user_messages_text for keyword in ["개인", "개인 개발자", "개인 사용자", "개인용", "개인으로", "팀", "팀용", "우리 팀"]) or  # 사용 형태가 있으면 충분
        "코딩" in all_user_messages_text or  # "코딩" 키워드가 있으면 충분
        "ai" in all_user_messages_text or  # "AI" 키워드가 있으면 충분
        "도구" in all_user_messages_text  # "도구" 키워드가 있으면 충분 (일반 추천 가능)
    )
    
    if not has_sufficient_info:
        print(f"⚡ [Decision Engine] 정보 부족 (너무 모호) - 빠른 반환 (team_size: {team_size}, budget_max: {budget_max}, dev_area: {has_development_area}, is_too_vague: {is_too_vague})")
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
        
        # 🚨 Follow-up 질문인 경우 이전 추천 순서 유지
        previous_tools_ordered = state.get("previous_tools_ordered")
        decision_result_dict = decision_result.model_dump()
        
        if previous_tools_ordered and len(previous_tools_ordered) > 0:
            print(f"🔍 [Decision Engine] 이전 추천 순서 확인: {previous_tools_ordered}")
            
            # 이전 순서를 기준으로 추천 도구 재정렬
            recommended_tools = decision_result.recommended_tools
            reordered_tools = []
            used_tools = set()
            
            # 1. 이전 순서대로 우선 배치 (현재 추천에 있는 도구만)
            for prev_tool in previous_tools_ordered:
                # 도구명 매칭 (대소문자 무시, 약간의 변형 허용)
                matched_tool = None
                for tool in recommended_tools:
                    prev_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', prev_tool.lower()).strip()
                    tool_clean = re.sub(r'[\(\)\[\]월\s\$0-9]+', '', tool.lower()).strip()
                    if prev_clean in tool_clean or tool_clean in prev_clean or tool_clean == prev_clean:
                        if tool not in used_tools:
                            matched_tool = tool
                            break
                
                if matched_tool:
                    reordered_tools.append(matched_tool)
                    used_tools.add(matched_tool)
            
            # 2. 이전 순서에 없는 도구들 추가 (하지만 Follow-up이면 이전 도구만 사용하는 것이 원칙)
            for tool in recommended_tools:
                if tool not in used_tools:
                    # Follow-up 질문이면 이전 도구만 사용하는 것이 원칙이지만,
                    # Decision Engine 결과가 있으면 최대한 활용
                    reordered_tools.append(tool)
                    used_tools.add(tool)
            
            # 재정렬된 도구 목록으로 DecisionResult 업데이트
            if reordered_tools:
                decision_result_dict["recommended_tools"] = reordered_tools
                print(f"✅ [Decision Engine] 이전 순서 적용: {reordered_tools}")
        
        return {
            "decision_result": decision_result_dict,
            "tool_facts": tool_facts  # tool_facts를 state에 저장하여 route_after_research에서 사용 가능하도록
        }
    except Exception as e:
        print(f"⚠️ [Decision Engine] 오류: {e}")
        import traceback
        traceback.print_exc()
        return {}

