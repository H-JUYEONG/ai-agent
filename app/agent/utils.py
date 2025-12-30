"""Utility functions for AI Service Advisor"""

import os
from typing import List
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field


class ThinkTool(BaseModel):
    """전략적 사고 도구"""
    reflection: str = Field(description="현재 상황에 대한 분석과 다음 단계 계획")


think_tool = ThinkTool


def get_api_key_for_model(model: str, config: dict) -> str:
    """모델에 맞는 API 키 반환"""
    return os.getenv("OPENAI_API_KEY", "")


def get_today_str() -> str:
    """오늘 날짜"""
    from datetime import datetime
    return datetime.now().strftime("%Y년 %m월 %d일")


def get_notes_from_tool_calls(messages: List) -> List[str]:
    """메시지에서 연구 노트 추출"""
    notes = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # ConductResearch 결과 또는 compressed_research 내용 추출
            if msg.name == "ConductResearch":
                content = str(msg.content)
                # "연구 실패"가 아닌 실제 내용만 추가
                if content and content != "연구 실패" and len(content.strip()) > 10:
                    notes.append(content)
    return notes


def get_buffer_string(messages: List) -> str:
    """메시지 리스트를 문자열로 변환"""
    result = []
    for msg in messages:
        if hasattr(msg, 'type'):
            result.append(f"{msg.type}: {msg.content}")
        else:
            result.append(str(msg))
    return "\n".join(result)



