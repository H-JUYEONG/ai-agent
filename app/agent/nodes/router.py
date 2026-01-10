"""라우팅 관련 노드 - clarify_with_user, route_after_research"""

import re
from typing import Literal

from app.agent.nodes._common import (
    Command,
    END,
    RunnableConfig,
    AgentState,
    ClarifyWithUser,
    Configuration,
    AIMessage,
    HumanMessage,
    get_buffer_string,
    configurable_model,
    clarify_with_user_instructions,
    get_today_str,
    get_api_key_for_model,
    query_normalizer,
    research_cache,
    vector_store,
)
from app.agent.nodes.writer import generate_greeting_dynamically


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", END]]:
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
    
    last_user_message = messages[-1].content if messages else ""
    
    # ========== 🚨 LLM 기반 주제 검증 및 인사 감지 (검색/캐시 전에 먼저 수행) ==========
    # 주제 검증을 LLM이 판단하도록 하여 불필요한 쿼리 정규화/캐시 조회 방지
    # 키워드 선검증 제거: LLM이 모든 질문의 주제 관련성을 판단
    model_config_clarify = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config_clarify)
    )
    
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str(),
        domain=domain,
        is_followup="YES" if is_followup else "NO"
    )
    
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 🆕 인사 메시지 체크 (가장 먼저!)
    if response.is_greeting:
        print(f"👋 [인사 응답] LLM이 인사 메시지 감지 - 친절하게 응답")
        # LLM이 greeting_message를 생성하도록 프롬프트에서 명시했으므로, 없으면 경고하고 fallback 사용
        greeting_msg = response.greeting_message
        if not greeting_msg or greeting_msg.strip() == "":
            print(f"⚠️ [인사 응답] LLM이 greeting_message를 생성하지 않음 - fallback 사용")
            greeting_msg = (
                "안녕하세요! 반갑습니다 😊\n\n"
                "저는 코딩 AI 도구 추천을 전문으로 하는 어시스턴트입니다. "
                "팀에 적합한 코딩 AI 도구(코드 작성, 리뷰, 자동 완성 등)에 대해 궁금한 점이 있으면 언제든지 물어보세요!"
            )
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=greeting_msg)]}
        )
    
    # 🚨 주제 관련성 체크 (검색/캐시 전 차단)
    if not response.is_on_topic:
        print(f"⚠️ [주제 검증] 주제에서 벗어난 질문 감지 - 캐시/검색/벡터DB 저장 차단")
        off_topic_msg = response.off_topic_message if response.off_topic_message else "죄송합니다. 저는 코딩 AI 도구 추천을 전문으로 하는 어시스턴트입니다. 다시 말씀해주세요!"
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=off_topic_msg)]}
        )
    
    # 주제 검증 통과 → 이제 쿼리 정규화 및 캐시 조회 진행
    print(f"✅ [주제 검증] 주제 검증 통과 - 정상 프로세스 진행")
    
    # ========== 🆕 답변 형식 요청 감지 ==========
    # 사용자가 요청한 답변 형식 감지 (표, 테이블, 리스트 등)
    response_format = None
    last_user_message_lower = last_user_message.lower()
    
    if any(keyword in last_user_message_lower for keyword in ["표로 정리", "표로", "테이블로", "비교표", "표 형식", "표 형식으로"]):
        response_format = "table"
        print(f"🔍 [답변 형식] 테이블 형식 요청 감지")
    elif any(keyword in last_user_message_lower for keyword in ["리스트로", "목록으로", "리스트 형식", "목록 형식"]):
        response_format = "list"
        print(f"🔍 [답변 형식] 리스트 형식 요청 감지")
    else:
        response_format = "markdown"  # 기본값
        print(f"🔍 [답변 형식] 마크다운 형식 (기본값)")
    
    # ========== 🆕 1단계: 쿼리 정규화 (캐시 키 생성) ==========
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
                # 🚨 인사 멘트는 항상 LLM으로 동적 생성 (공통 함수 사용)
                print(f"✅ [캐시 처리] 리포트 본문은 캐시에서 가져옴 ({len(report_body)}자), 인사 멘트는 LLM으로 동적 생성")
                
                greeting = await generate_greeting_dynamically(messages, config, is_followup)
                if not greeting or len(greeting) < 20:
                    # LLM 생성 실패 시 질문 기반 최소 생성
                    if last_user_message:
                        greeting = f"{last_user_message[:50]}에 대해 분석해드리겠습니다."
                    else:
                        greeting = "분석해드리겠습니다."
                    print(f"⚠️ [캐시 처리] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
                print(f"✅ [캐시 처리] 리포트 본문 길이: {len(report_body)}자, 시작 100자: {report_body[:100]}")
                
                return Command(
                    goto=END,
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
                # 인사 멘트 생성 (LLM으로 동적 생성)
                print(f"✅ [유사 질문 처리] 리포트 본문은 캐시에서 가져옴 ({len(report_body)}자), 인사 멘트는 LLM으로 동적 생성")
                
                greeting = await generate_greeting_dynamically(messages, config, is_followup)
                if not greeting or len(greeting) < 20:
                    # LLM 생성 실패 시 질문 기반 최소 생성
                    if last_user_message:
                        greeting = f"{last_user_message[:50]}에 대해 분석해드리겠습니다."
                    else:
                        greeting = "분석해드리겠습니다."
                    print(f"⚠️ [유사 질문 처리] LLM 멘트 생성 실패 또는 너무 짧음, fallback 사용: '{greeting}'")
                
                return Command(
                    goto=END,
                    update={"messages": [
                        AIMessage(content=greeting),
                        AIMessage(content=report_body)
                    ]}
                )
    
    # 캐시 미스 및 유사 질문도 없음 → 새로 생성
    # 주제 검증은 이미 위(라인 138-147)에서 완료되었으므로 response를 재사용
    print(f"⚠️ [캐시 MISS + 유사 질문 없음] 새로 생성 진행 (주제 검증 완료)")
    
    # 🆕 검색 필요 여부 체크 (Follow-up 질문인 경우)
    need_research = getattr(response, 'need_research', True)  # 기본값: True (검색 필요)
    if not need_research:
        print(f"✅ [검색 불필요] 이전 대화 정보만으로 답변 가능 - 검색 건너뛰고 바로 리포트 생성")
        # 검색 없이 바로 final_report_generation으로 이동
        return Command(
            goto="final_report_generation",
            update={
                "messages": [AIMessage(content=response.verification if response.verification else "네! 이전 추천 내용을 정리해드리겠습니다.")],
                "normalized_query": normalized,  # 🆕 정규화 정보 저장
                "response_format": response_format,  # 🆕 답변 형식 저장
                "need_research": False  # 🆕 재검색 불필요 플래그 저장 (캐시/벡터 DB 저장 건너뛰기용)
            }
        )
        print(f"✅ [검색 불필요] state에 response_format='{response_format}' 저장 완료")
    
    # 명확화 비활성화 시 바로 다음 단계로 (주제 검증은 이미 완료됨)
    if not configurable.allow_clarification:
        print(f"✅ [DEBUG] 주제 검증 통과 - 바로 연구 시작")
        print(f"✅ [명확화 비활성화] state에 response_format='{response_format}' 저장 완료")
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "normalized_query": normalized,  # 🆕 정규화 정보 저장
                "response_format": response_format,  # 🆕 답변 형식 저장
                "need_research": True  # 🆕 검색 필요 플래그 저장
            }
        )
    
    # 명확화 필요 여부 체크
    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        print(f"✅ [검색 필요] state에 response_format='{response_format}' 저장 완료")
        return Command(
            goto="write_research_brief",
            update={
                "messages": [AIMessage(content=response.verification)],
                "normalized_query": normalized,  # 🆕 정규화 정보 저장
                "response_format": response_format,  # 🆕 답변 형식 저장
                "need_research": True  # 🆕 검색 필요 플래그 저장
            }
        )


def route_after_research(state: AgentState) -> Literal["structured_report_generation", "final_report_generation", "clarify_missing_constraints", "cannot_answer"]:
    """연구 완료 후 라우팅: Decision Engine 결과 유무와 제약 조건 충분 여부에 따라 분기"""
    
    import re
    
    # Decision Engine이 실행되어야 하는 질문인지 확인
    question_type = state.get("question_type", "comparison")
    messages_list = state.get("messages", [])
    
    # 🚨 HumanMessage만 추출 (AI 응답 메시지 제외)
    human_messages = [msg for msg in messages_list if isinstance(msg, HumanMessage)]
    all_user_messages_text = " ".join([str(msg.content).lower() for msg in human_messages])
    last_user_message = str(human_messages[-1].content).lower() if human_messages else ""
    
    # 🚨 디버깅: 질문 내용과 타입 확인
    print(f"🔍 [Routing DEBUG] question_type: {question_type}")
    print(f"🔍 [Routing DEBUG] HumanMessage 개수: {len(human_messages)}")
    print(f"🔍 [Routing DEBUG] last_user_message: {last_user_message[:100] if last_user_message else 'None'}")
    
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
        ("최적화" in last_user_message and "도구" in last_user_message)
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
    
    # 🚨 전체 사용자 메시지 히스토리에서 정보 추출
    has_user_type = False
    if not team_size and all_user_messages_text:
        # "개인", "개인 개발자", "개인 사용자" 등을 인식하여 team_size = 1로 설정
        if any(keyword in all_user_messages_text for keyword in ["개인", "개인 개발자", "개인 사용자", "개인용", "개인으로"]):
            team_size = 1
            has_user_type = True
        elif any(keyword in all_user_messages_text for keyword in ["팀", "팀용", "우리 팀", "팀 규모"]):
            has_user_type = True
            # "X명" 패턴 찾기
            team_size_match = re.search(r'(\d+)\s*명', all_user_messages_text)
            if team_size_match:
                team_size = int(team_size_match.group(1))
        else:
            # "X명" 패턴 찾기
            team_size_match = re.search(r'(\d+)\s*명', all_user_messages_text)
            if team_size_match:
                team_size = int(team_size_match.group(1))
    
    # 개발 언어/분야 확인 (전체 메시지 히스토리에서)
    has_development_area = False
    if all_user_messages_text:
        # 프로그래밍 언어 확인 (하드코딩 - 언어는 정해져 있으므로 OK)
        languages = ["python", "javascript", "java", "typescript", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin", "dart", "r", "scala", "clojure", "perl", "lua", "matlab"]
        # 개발 분야 확인
        domains = ["웹 개발", "백엔드", "프론트엔드", "풀스택", "모바일", "게임", "데이터", "ai", "ml", "머신러닝", "앱 개발"]
        # 프레임워크/라이브러리 확인 (주요 프레임워크만)
        frameworks = ["react", "vue", "angular", "django", "flask", "spring", "node.js", "express", "fastapi", "laravel", "rails"]
        
        # "~으로 개발", "~로 개발", "~ 개발", "~을 사용" 같은 표현도 포함
        if any(lang in all_user_messages_text for lang in languages) or \
           any(domain in all_user_messages_text for domain in domains) or \
           any(fw in all_user_messages_text for fw in frameworks) or \
           re.search(r'으로\s*개발|로\s*개발|개발', all_user_messages_text):
            has_development_area = True
    
    # 🚨 매우 중요: 기본적으로 정보가 충분하다고 가정!
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
        if has_vague_pattern and not has_development_area and not has_user_type and not team_size and not budget_max:
            is_too_vague = True
    
    # 정보 충분 여부 판단: 모호하지 않고, 어느 정도 정보가 있으면 충분
    has_sufficient_info = not is_too_vague and (
        has_development_area or  # 개발 언어/분야가 있으면 충분
        has_user_type or  # 사용 형태가 있으면 충분
        team_size is not None or  # 팀 규모가 있으면 충분
        budget_max is not None or  # 예산이 있으면 충분
        "코딩" in all_user_messages_text or  # "코딩" 키워드가 있으면 충분
        "ai" in all_user_messages_text or  # "AI" 키워드가 있으면 충분
        "도구" in all_user_messages_text  # "도구" 키워드가 있으면 충분 (일반 추천 가능)
    )
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
    print(f"🔍 [Routing DEBUG] 정보 충분 여부: {has_sufficient_info} (user_type: {has_user_type}, dev_area: {has_development_area}, team_size: {team_size}, budget_max: {budget_max})")
    
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
        elif not has_sufficient_info:
            # 🚨 개발 언어/분야도 없고 제약 조건도 없으면 명확화 필요
            print(f"🔍 [Routing] Decision 질문이지만 정보 부족 (user_type: {has_user_type}, dev_area: {has_development_area}, team_size: {team_size}, budget: {budget_max}) → clarify_missing_constraints")
            return "clarify_missing_constraints"
        else:
            # 🚨 제약 조건은 없지만 개발 언어/분야가 있으면 충분한 정보!
            # Decision Engine 결과가 없어도 일반 리포트로 추천 제공
            print(f"✅ [Routing] Decision 질문 + 개발 언어/분야 정보 있음 (제약 조건 없지만 충분) → final_report_generation")
            return "final_report_generation"
    else:
        # Discovery 질문인 경우: 일반 리포트 생성 (Decision Engine 불필요)
        print(f"✅ [Routing] Discovery 질문 → final_report_generation")
        return "final_report_generation"

