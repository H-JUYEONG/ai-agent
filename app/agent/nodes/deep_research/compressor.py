"""연구 결과 압축 노드 - compress_research"""

from app.agent.nodes._common import (
    RunnableConfig,
    ResearcherState,
    Configuration,
    HumanMessage,
    SystemMessage,
    configurable_model,
    compress_research_system_prompt,
    compress_research_simple_human_message,
    get_today_str,
    get_api_key_for_model,
    ToolMessage,
    AIMessage,
)


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

