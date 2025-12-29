"""LangGraph workflow for AI Service Advisor"""

from langgraph.graph import END, START, StateGraph

from app.agent.configuration import Configuration
from app.agent.state import (
    AgentInputState,
    AgentState,
    ResearcherOutputState,
    ResearcherState,
    SupervisorState,
)
from app.agent.nodes import (
    clarify_with_user,
    write_research_brief,
    supervisor,
    supervisor_tools,
    researcher,
    researcher_tools,
    compress_research,
    final_report_generation,
)


# Researcher Subgraph (개별 연구원 워크플로우)
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)

researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)

researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)

researcher_subgraph = researcher_builder.compile()


# Supervisor Subgraph (연구 관리 워크플로우)
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")

supervisor_subgraph = supervisor_builder.compile()


# Main Deep Researcher Graph (전체 워크플로우)
deep_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 워크플로우 연결
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 메인 그래프 컴파일
deep_researcher = deep_researcher_builder.compile()



