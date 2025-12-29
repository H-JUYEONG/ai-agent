"""LangGraph nodes for AI Service Advisor"""

import asyncio
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
    
    # 간단한 판단: Messages 개수만으로
    is_followup = len(messages) >= 3  # 질문1 + 답변1 + 질문2
    
    # 디버깅
    print(f"🔍 [DEBUG] clarify - Messages: {len(messages)}개, Follow-up: {is_followup}")
    
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
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=cached_answer["content"])]}
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
    is_followup = len(messages_list) >= 3
    
    # 이전 도구 추출 (Follow-up인 경우)
    previous_tools = ""
    if is_followup:
        import re
        for msg in reversed(messages_list[:-1]):
            if hasattr(msg, 'content'):
                tools_found = re.findall(r'📊\s+([^\n]+)', str(msg.content))
                if tools_found:
                    previous_tools = ", ".join(tools_found[:10])
                    break
    
    # 질문 유형 판단
    last_user_msg = messages_list[-1].content.lower() if messages_list else ""
    question_type = "comparison"  # 기본값
    
    if any(kw in last_user_msg for kw in ["하나만", "최종", "결정", "선택", "골라"]):
        question_type = "decision"
    elif any(kw in last_user_msg for kw in ["왜", "이유", "차이", "포기", "설명", "뭐가 달라"]):
        question_type = "explanation"
    elif any(kw in last_user_msg for kw in ["가격", "얼마", "비용", "어떤 기능", "선호도", "인기도", "많이 사용", "더 많이", "사람들이", "평가"]):
        question_type = "information"
    
    print(f"🔍 [DEBUG] write_research_brief - Messages: {len(messages_list)}개, Follow-up: {is_followup}, 질문유형: {question_type}, 이전 도구: {previous_tools}")
    
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(messages_list),
        date=get_today_str(),
        current_year=get_current_year(),
        current_month_year=get_current_month_year(),
        domain=domain,
        domain_guide=formatted_domain_guide_for_research,
        is_followup="YES" if is_followup else "NO",
        previous_tools=previous_tools if previous_tools else "없음",
        question_type=question_type
    )
    
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # 디버깅: Research Brief 확인
    print(f"🔍 [DEBUG] Research Brief: {response.research_brief[:200]}...")
    
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
        return Command(
            goto="__end__",
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
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
        raw_notes = "\n".join([
            "\n".join(obs.get("raw_notes", [])) for obs in results
        ])
        if raw_notes:
            update_payload["raw_notes"] = [raw_notes]
    
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
    
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
    }
    
    # Messages 가져오기 및 Follow-up 판단
    messages_list = state.get("messages", [])
    is_followup = len(messages_list) >= 3
    
    # 이전 도구 추출 (Follow-up인 경우)
    previous_tools = ""
    if is_followup:
        import re
        for msg in reversed(messages_list[:-1]):
            if hasattr(msg, 'content'):
                tools_found = re.findall(r'📊\s+([^\n]+)', str(msg.content))
                if tools_found:
                    previous_tools = ", ".join(tools_found[:10])
                    break
    
    # 질문 유형 판단
    last_user_msg = messages_list[-1].content.lower() if messages_list else ""
    question_type = "comparison"  # 기본값
    
    if any(kw in last_user_msg for kw in ["하나만", "최종", "결정", "선택", "골라"]):
        question_type = "decision"
    elif any(kw in last_user_msg for kw in ["왜", "이유", "차이", "포기", "설명", "뭐가 달라"]):
        question_type = "explanation"
    elif any(kw in last_user_msg for kw in ["가격", "얼마", "비용", "어떤 기능", "선호도", "인기도", "많이 사용", "더 많이", "사람들이", "평가"]):
        question_type = "information"
    
    print(f"🔍 [DEBUG] final_report - Messages: {len(messages_list)}개, Follow-up: {is_followup}, 질문유형: {question_type}, 이전 도구: {previous_tools}")
    
    final_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(messages_list),
        findings=findings,
        date=get_today_str(),
        is_followup="YES" if is_followup else "NO",
        previous_tools=previous_tools if previous_tools else "없음",
        question_type=question_type
    )
    
    try:
        final_report = await configurable_model.with_config(writer_model_config).ainvoke([
            HumanMessage(content=final_prompt)
        ])
        
        report_content = str(final_report.content)
        
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
            import re
            # 태그와 내용을 추출 (여러 줄 포함)
            match = re.search(r'\[GREETING\](.*?)\[/GREETING\]', report_content, re.DOTALL)
            if match:
                greeting = match.group(1).strip()
                # 태그 전체를 제거하고 나머지를 리포트로
                report_body = report_content.replace(match.group(0), "").strip()
                
                print(f"✅ [DEBUG] 인사말 추출 성공: {greeting[:50]}...")
                print(f"✅ [DEBUG] 리포트 본문 시작: {report_body[:100]}")
                
                # 두 개의 메시지로 반환
                messages_to_add = [
                    AIMessage(content=greeting),
                    AIMessage(content=report_body)
                ]
            else:
                print(f"❌ [DEBUG] 태그 파싱 실패 - 정규식 매칭 실패")
                # 태그 파싱 실패 시 전체를 하나의 메시지로
                messages_to_add = [final_report]
        else:
            print(f"✅ [DEBUG] GREETING 태그 없음 - 일반 리포트")
            # 인사말 없는 경우 하나의 메시지로
            messages_to_add = [final_report]
        
        return {
            "final_report": report_content,
            "messages": messages_to_add,
            "notes": {"type": "override", "value": []}
        }
    
    except Exception as e:
        print(f"❌ 리포트 생성 실패: {e}")
        return {
            "final_report": f"리포트 생성 중 오류 발생: {str(e)}",
            "messages": [AIMessage(content="리포트 생성 실패")],
            "notes": {"type": "override", "value": []}
        }



