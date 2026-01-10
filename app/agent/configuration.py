"""Configuration for AI Service Advisor"""

import os
from typing import Optional, Any
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """AI Service Advisor 설정"""
    
    # 일반 설정
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=False)  # 빠른 시작을 위해 False
    max_concurrent_research_units: int = Field(default=3)  # 병렬 연구 최대 3개 (속도 향상!)
    
    # 연구 설정
    max_researcher_iterations: int = Field(default=1)  # 1회 반복
    max_react_tool_calls: int = Field(default=3)  # 각 researcher 3번 검색 (속도 향상!)
    
    # 모델 설정 (비용 최적화 - 모두 mini 사용!)
    research_model: str = Field(default="gpt-4o-mini")
    research_model_max_tokens: int = Field(default=8192)
    
    compression_model: str = Field(default="gpt-4o-mini")
    compression_model_max_tokens: int = Field(default=8192)
    
    final_report_model: str = Field(default="gpt-4o-mini")  # 비용 절감!
    final_report_model_max_tokens: int = Field(default=16384)  # gpt-4o-mini 최대 토큰 제한 (16384)
    
    # 검색 설정
    search_max_results: int = Field(default=5)  # 결과 5개 (정확도↑)
    search_depth: str = Field(default="advanced")  # advanced (정확도↑)
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """RunnableConfig에서 Configuration 생성"""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
    
    class Config:
        arbitrary_types_allowed = True



