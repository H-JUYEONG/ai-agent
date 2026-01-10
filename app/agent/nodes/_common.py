"""공통 import 및 설정"""

import asyncio
import re
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
from langgraph.graph import END

from app.agent.configuration import Configuration
from app.agent.state import (
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchQuestion,
    ResearcherState,
    SupervisorState,
    TableData,
)
from app.agent.models import ToolFact, UserContext, PricingPlan, SecurityPolicy, WorkflowType
from app.agent.decision import DecisionEngine
from app.agent.fact_extractor import extract_tool_facts
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

