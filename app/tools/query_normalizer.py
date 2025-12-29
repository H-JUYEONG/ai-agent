"""LLM 기반 쿼리 정규화 - 의미적으로 유사한 질문을 같은 키로 변환"""

import hashlib
from typing import Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class NormalizedQuery(BaseModel):
    """정규화된 쿼리"""
    normalized_text: str = Field(
        description="정규화된 질문 (핵심 의도만 추출, 동일한 의미는 동일한 표현)"
    )
    keywords: list[str] = Field(
        description="핵심 키워드 리스트 (도구명, 기능 등)",
        default_factory=list
    )
    intent: str = Field(
        description="질문 의도 (comparison, pricing, features, recommendation 등)",
        default="general"
    )


NORMALIZATION_PROMPT = """당신은 사용자 질문을 정규화하는 전문가입니다.

**목표:** 의미적으로 동일한 질문들을 같은 형태로 변환하여, 캐시 히트율을 높입니다.

**규칙:**
1. 존댓말/반말, 문체 차이 무시 → 표준 형태로 통일
2. 불필요한 수식어, 감탄사 제거
3. 핵심 키워드 추출 (도구명, 작업 유형, 팀 규모 등)
4. 질문 의도 파악 (비교, 가격, 기능, 추천 등)
5. **중요**: 동일한 의미의 질문은 반드시 동일한 normalized_text와 keywords를 생성해야 함

**예시:**
- "PyCharm 쓰는 데이터 분석 팀입니다. Python 위주 작업에 잘 맞고, 팀 단위로 쓰기 좋은 코딩 AI 추천해주세요."
  → normalized_text: "Python 데이터 분석 팀 코딩 AI 추천"
  → keywords: ["Python", "데이터 분석", "팀", "코딩 AI"]

- "PyCharm 쓰는 데이터 분석 팀인데 팀 단위로 쓰기 좋은 코딩 AI 추천해줘"
  → normalized_text: "Python 데이터 분석 팀 코딩 AI 추천"  (위와 동일)
  → keywords: ["Python", "데이터 분석", "팀", "코딩 AI"]  (위와 동일)

- "Copilot 회사에서 써도 괜찮아요?" → "Copilot 기업 사용 가능 여부"
- "깃헙 코파일럿 보안 문제 없나요?" → "Copilot 기업 사용 가능 여부"
- "Cursor랑 Copilot 뭐가 더 좋아?" → "Cursor Copilot 비교"
- "커서 vs 코파일럿 어떤거 써야해?" → "Cursor Copilot 비교"
- "Cursor 가격이 어떻게 되나요?" → "Cursor 가격 정보"
- "커서 얼마에요?" → "Cursor 가격 정보"

**입력:** {user_query}

**출력 형식:**
- normalized_text: 정규화된 질문 (명확하고 간결하게, 동일한 의미는 동일한 텍스트)
- keywords: [도구명, 작업유형, 팀규모 등] (정렬된 순서로, 동일한 키워드면 동일한 리스트)
- intent: 질문 의도 (comparison / pricing / features / recommendation / security / general)
"""


class QueryNormalizer:
    """쿼리 정규화 - 캐시 히트율 향상"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 200):
        """
        Args:
            model: 사용할 LLM 모델
            max_tokens: 최대 토큰 수
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.configurable_model = init_chat_model(
            configurable_fields=("model", "max_tokens", "api_key"),
        )
    
    async def normalize(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        쿼리 정규화
        
        Args:
            query: 사용자 질문
            config: LangChain RunnableConfig (API 키 등)
        
        Returns:
            {
                "normalized_text": "정규화된 질문",
                "keywords": ["키워드1", "키워드2"],
                "intent": "comparison",
                "cache_key": "해시키"
            }
        """
        try:
            # LLM 호출 설정
            model_config = config or {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
            }
            
            normalizer_model = (
                self.configurable_model
                .with_structured_output(NormalizedQuery)
                .with_config(model_config)
            )
            
            # 정규화 실행
            prompt = NORMALIZATION_PROMPT.format(user_query=query)
            response = await normalizer_model.ainvoke([HumanMessage(content=prompt)])
            
            # 캐시 키 생성 (정규화된 텍스트 + 정렬된 키워드 기반)
            # 키워드를 정렬하고 소문자로 통일하여 일관성 보장
            sorted_keywords = sorted([kw.lower().strip() for kw in response.keywords if kw])
            normalized_text_lower = response.normalized_text.lower().strip()
            cache_key_str = f"{normalized_text_lower}:{':'.join(sorted_keywords)}"
            cache_key = hashlib.md5(cache_key_str.encode()).hexdigest()
            
            result = {
                "normalized_text": response.normalized_text,
                "keywords": response.keywords,
                "intent": response.intent,
                "cache_key": cache_key
            }
            
            print(f"✅ 쿼리 정규화: '{query[:50]}...' → '{response.normalized_text}' (캐시키: {cache_key[:8]}...)")
            return result
        
        except Exception as e:
            print(f"⚠️ 쿼리 정규화 실패 (Fallback: 원본 사용): {e}")
            # Fallback: 원본 쿼리 사용
            cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
            return {
                "normalized_text": query,
                "keywords": [],
                "intent": "general",
                "cache_key": cache_key
            }


# 전역 Normalizer 인스턴스
query_normalizer = QueryNormalizer()

