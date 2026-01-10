"""연구원 노드 - researcher, researcher_tools"""

import re
from datetime import datetime
from typing import Literal

from app.agent.nodes._common import (
    Command,
    RunnableConfig,
    ResearcherState,
    Configuration,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    configurable_model,
    DOMAIN_GUIDES,
    research_system_prompt,
    get_today_str,
    get_current_year,
    get_current_month_year,
    get_api_key_for_model,
    think_tool,
    searcher,
    vector_store,
)


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
                        age_days = (datetime.now().timestamp() - fact['created_at']) / 86400
                        formatted += f"{idx}. [신뢰도 {fact['score']:.2f}, {age_days:.0f}일 전]\n"
                        formatted += f"   {fact['text'][:300]}...\n"
                        formatted += f"   출처: {fact['source']} ({fact.get('url', '')[:50]}...)\n\n"
                    content = formatted
                else:
                    # 결과가 부족하면 웹 검색 필요
                    formatted = f"⚠️ Vector DB에서 {len(facts)}개 관련 정보 발견 (부족함, 웹 검색 필요):\n\n"
                    for idx, fact in enumerate(facts, 1):
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

