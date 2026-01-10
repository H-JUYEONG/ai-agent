"""연구 슈퍼바이저 노드 - supervisor, supervisor_tools"""

import asyncio
from typing import Literal

from app.agent.nodes._common import (
    Command,
    END,
    RunnableConfig,
    SupervisorState,
    Configuration,
    HumanMessage,
    ToolMessage,
    configurable_model,
    ConductResearch,
    ResearchComplete,
    think_tool,
    get_api_key_for_model,
    get_notes_from_tool_calls,
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

