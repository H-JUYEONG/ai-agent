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
    is_on_topic: bool = Field(
        description="사용자 질문이 코딩 AI 도구 추천과 관련 있는가?",
    )
    need_clarification: bool = Field(
        description="사용자에게 추가 질문이 필요한가?",
        default=False,
    )
    question: str = Field(
        description="사용자에게 물어볼 명확화 질문 (명확화가 필요한 경우)",
        default="",
    )
    verification: str = Field(
        description="연구 시작 확인 메시지 (주제가 맞고 명확화가 불필요한 경우)",
        default="",
    )
    off_topic_message: str = Field(
        description="주제에서 벗어난 경우 사용자에게 보낼 거부 메시지",
        default="죄송합니다. 저는 코딩 AI 도구 추천을 전문으로 하는 챗봇입니다. 다시 질문해주세요!",
    )


class HardConstraints(BaseModel):
    """하드 제약 조건 (필터링 규칙)"""
    budget_max: Optional[int] = Field(
        default=None,
        description="최대 예산 (원화 기준)"
    )
    security_required: bool = Field(
        default=False,
        description="보안/프라이버시 필수 여부"
    )
    excluded_tools: List[str] = Field(
        default_factory=list,
        description="명시적으로 제외할 도구 목록"
    )
    excluded_features: List[str] = Field(
        default_factory=list,
        description="금지된 기능 목록 (예: 외부 서버 전송, 클라우드 저장)"
    )
    team_size: Optional[int] = Field(
        default=None,
        description="팀 규모"
    )
    must_support_ide: List[str] = Field(
        default_factory=list,
        description="반드시 지원해야 할 IDE"
    )
    must_support_language: List[str] = Field(
        default_factory=list,
        description="반드시 지원해야 할 언어"
    )
    other_requirements: List[str] = Field(
        default_factory=list,
        description="기타 요구사항"
    )


class ResearchQuestion(BaseModel):
    """연구 질문 생성"""
    research_brief: str = Field(
        description="연구를 안내할 구체적인 연구 질문",
    )
    question_type: str = Field(
        description="""사용자 질문의 의도와 유형을 분석하여 분류:
- "decision": 의사결정 요청 (1순위, 2순위, 최종 선택 등)
- "comparison": 비교/평가 요청 (특히 강한, 더 좋은, 어떤 도구 등)
- "explanation": 설명 요청 (왜, 이유, 차이 등)
- "information": 정보 요청 (가격, 기능, 선호도 등)
- "guide": 실무 가이드 요청 (설정 방법, 도입 가이드, 사용법, 세팅 방법, 주의사항 등)

**중요**: "설정", "세팅", "방법", "가이드", "도입", "사용법", "어떻게", "주의사항" 같은 표현이 있으면 "guide"로 분류하세요.

사용자 질문의 핵심 의도를 정확히 파악하여 분류하세요.""",
        default="comparison"
    )
    hard_constraints: HardConstraints = Field(
        default_factory=HardConstraints,
        description="""사용자가 명시한 하드 제약 조건 (필터링 규칙).
        
**중요**: "금지", "불가", "차단", "제외", "안 됨", "절대 안 됨", "사용할 수 없음" 같은 표현이 있으면 
해당 항목을 excluded_tools 또는 excluded_features에 반드시 추가하세요."""
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
    normalized_query: Optional[dict] = None  # 정규화된 쿼리 정보
    constraints: Optional[dict] = None  # 하드 제약 조건 (필터링 규칙)
    question_type: Optional[str] = None  # LLM이 판단한 질문 유형


class SupervisorState(TypedDict):
    """Supervisor 상태"""
    supervisor_messages: Annotated[List[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[List[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[List[str], override_reducer] = []
    domain: Optional[str]
    constraints: Optional[dict]  # 하드 제약 조건


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



