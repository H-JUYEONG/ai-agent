"""Prompts for AI Service Advisor - 모듈화된 구조"""

# 유틸리티 함수
from app.agent.prompts.utils import (
    get_today_str,
    get_current_year,
    get_current_month_year,
)

# 도메인 가이드
from app.agent.prompts.domain import DOMAIN_GUIDES

# 프롬프트 템플릿
from app.agent.prompts.clarify import clarify_with_user_instructions
from app.agent.prompts.research import (
    transform_messages_into_research_topic_prompt,
    lead_researcher_prompt,
    research_system_prompt,
)
from app.agent.prompts.compress import (
    compress_research_system_prompt,
    compress_research_simple_human_message,
)
from app.agent.prompts.report import final_report_generation_prompt

__all__ = [
    "get_today_str",
    "get_current_year",
    "get_current_month_year",
    "DOMAIN_GUIDES",
    "clarify_with_user_instructions",
    "transform_messages_into_research_topic_prompt",
    "lead_researcher_prompt",
    "research_system_prompt",
    "compress_research_system_prompt",
    "compress_research_simple_human_message",
    "final_report_generation_prompt",
]

