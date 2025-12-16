"""Graph state definitions for AI Service Advisor"""

import operator
from typing import Annotated, Optional, List
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################

class ConductResearch(BaseModel):
    """연구 수행 도구"""
    research_topic: str = Field(
        description="연구할 구체적인 주제. 상세하게 작성 (최소 1개 문단).",
    )


class ResearchComplete(BaseModel):
    """연구 완료 신호"""
    pass


class ClarifyWithUser(BaseModel):
    """사용자 질문 정제"""
    need_clarification: bool = Field(
        description="사용자에게 추가 질문이 필요한가?",
    )
    question: str = Field(
        description="사용자에게 물어볼 명확화 질문",
    )
    verification: str = Field(
        description="연구 시작 확인 메시지",
    )


class ResearchQuestion(BaseModel):
    """연구 질문 생성"""
    research_brief: str = Field(
        description="연구를 안내할 구체적인 연구 질문",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """상태값 오버라이드 가능한 reducer"""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """입력 상태 (messages만)"""
    pass


class AgentState(MessagesState):
    """메인 에이전트 상태"""
    supervisor_messages: Annotated[List[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[List[str], override_reducer] = []
    notes: Annotated[List[str], override_reducer] = []
    final_report: str
    domain: Optional[str]  # LLM, 코딩, 디자인


class SupervisorState(TypedDict):
    """Supervisor 상태"""
    supervisor_messages: Annotated[List[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[List[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[List[str], override_reducer] = []
    domain: Optional[str]


class ResearcherState(TypedDict):
    """Researcher 상태"""
    researcher_messages: Annotated[List[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], override_reducer] = []
    domain: Optional[str]


class ResearcherOutputState(BaseModel):
    """Researcher 출력"""
    compressed_research: str
    raw_notes: Annotated[List[str], override_reducer] = []



