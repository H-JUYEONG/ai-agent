"""Fact Models for structured tool information"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class SecurityPolicy(str, Enum):
    """보안 정책 타입"""
    OPT_IN = "opt-in"  # 기본적으로 코드 전송, 사용자가 선택적으로 차단
    OPT_OUT = "opt-out"  # 기본적으로 코드 전송, 사용자가 선택적으로 차단 가능
    ON_PREMISE = "on-premise"  # 온프레미스만
    NO_TRANSMISSION = "no-transmission"  # 코드 전송 없음


class WorkflowType(str, Enum):
    """업무 타입"""
    CODE_COMPLETION = "code_completion"  # 코드 자동완성
    CODE_GENERATION = "code_generation"  # 코드 생성
    CODE_REVIEW = "code_review"  # 코드 리뷰
    REFACTORING = "refactoring"  # 리팩토링
    DEBUGGING = "debugging"  # 디버깅
    DOCUMENTATION = "documentation"  # 문서화


class PricingPlan(BaseModel):
    """가격 플랜"""
    name: str = Field(description="플랜명 (⚠️ 예시: Free, Pro, Business, Enterprise는 참고용일 뿐, 실제 플랜명은 Findings에서 확인한 값 사용)")
    price_per_user_per_month: Optional[float] = Field(
        default=None,
        description="사용자당 월 가격 (USD)"
    )
    price_per_month: Optional[float] = Field(
        default=None,
        description="월 가격 (USD, 개인용)"
    )
    price_per_year: Optional[float] = Field(
        default=None,
        description="연간 가격 (USD, 전체 팀 또는 개인용)"
    )
    price_per_user_per_year: Optional[float] = Field(
        default=None,
        description="사용자당 연간 가격 (USD)"
    )
    plan_type: str = Field(
        description="플랜 타입: individual, team, enterprise, usage-based"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="가격 정보 출처 URL"
    )


class ToolFact(BaseModel):
    """도구 사실 (Fact Model)"""
    name: str = Field(description="도구명")
    
    # 가격 정보
    pricing_plans: List[PricingPlan] = Field(
        default_factory=list,
        description="가격 플랜 목록"
    )
    
    # 통합 기능
    integrations: List[str] = Field(
        default_factory=list,
        description="통합 서비스 목록 (예: GitHub, GitLab, Slack, Jira)"
    )
    
    # 언어 지원
    supported_languages: List[str] = Field(
        default_factory=list,
        description="지원하는 프로그래밍 언어 목록"
    )
    
    # 보안 정책
    security_policy: Optional[SecurityPolicy] = Field(
        default=None,
        description="보안 정책"
    )
    security_details: Optional[str] = Field(
        default=None,
        description="보안 정책 상세 설명"
    )
    
    # 업무 적합성
    workflow_support: List[WorkflowType] = Field(
        default_factory=list,
        description="지원하는 업무 타입"
    )
    
    # 주요 기능
    primary_features: List[str] = Field(
        default_factory=list,
        description="주요 기능 목록"
    )
    
    # 기능 카테고리 (중복 검사용)
    feature_category: str = Field(
        default="code_completion",
        description="기능 카테고리: code_completion, code_review, security_scan 등"
    )
    
    # 출처
    source_urls: List[str] = Field(
        default_factory=list,
        description="정보 출처 URL 목록"
    )
    
    # 메타 정보
    last_updated: Optional[str] = Field(
        default=None,
        description="마지막 업데이트 날짜"
    )


class UserContext(BaseModel):
    """사용자 맥락"""
    team_size: Optional[int] = Field(
        default=None,
        description="팀 규모"
    )
    tech_stack: List[str] = Field(
        default_factory=list,
        description="기술 스택 (언어, 프레임워크)"
    )
    budget_max: Optional[float] = Field(
        default=None,
        description="최대 예산 (USD/월)"
    )
    security_required: bool = Field(
        default=False,
        description="보안 필수 여부"
    )
    required_integrations: List[str] = Field(
        default_factory=list,
        description="필수 통합 서비스"
    )
    workflow_focus: List[WorkflowType] = Field(
        default_factory=list,
        description="주요 업무 타입"
    )
    excluded_tools: List[str] = Field(
        default_factory=list,
        description="제외할 도구 목록"
    )


class ToolScore(BaseModel):
    """도구 점수"""
    tool_name: str
    total_score: float = Field(description="총점 (0.0 ~ 1.0)")
    language_support_score: float = Field(description="언어 지원 점수")
    integration_score: float = Field(description="통합 기능 점수")
    workflow_fit_score: float = Field(description="업무 적합성 점수")
    price_score: float = Field(description="가격 점수")
    security_score: float = Field(description="보안 점수")
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="제외된 이유 (점수가 0이면)"
    )


class DecisionResult(BaseModel):
    """판단 결과"""
    recommended_tools: List[str] = Field(
        description="추천 도구 목록 (점수 순)"
    )
    excluded_tools: List[str] = Field(
        default_factory=list,
        description="제외된 도구 목록"
    )
    tool_scores: List[ToolScore] = Field(
        description="도구별 점수"
    )
    reasoning: Dict[str, str] = Field(
        description="각 도구에 대한 판단 이유"
    )

