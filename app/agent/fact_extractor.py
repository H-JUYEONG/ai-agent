"""Fact Extraction from Research Findings"""

import json
import re
from typing import List, Dict, Optional
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agent.models import ToolFact, PricingPlan, SecurityPolicy, WorkflowType
from app.agent.utils import get_api_key_for_model
from app.agent.configuration import Configuration


fact_extraction_prompt = """당신은 연구 결과에서 구조화된 사실을 추출하는 전문가입니다.

연구 결과(Findings)에서 각 도구에 대한 다음 정보를 추출하여 JSON 형식으로 반환하세요:

**추출할 정보:**
1. **도구명** (name)
2. **가격 플랜** (pricing_plans):
   - 🚨 **중요**: 개인용 플랜과 팀용 플랜을 반드시 구분하여 추출하세요!
   - 플랜명 (name): Findings에서 확인한 실제 플랜명 사용 (예: "Free", "Pro", "Business", "Team", "Enterprise" 등)
   - **개인용 플랜** (plan_type: "individual"):
     * price_per_month: 개인 사용자 월 가격 (USD)
     * price_per_user_per_month: null
   - **팀용 플랜** (plan_type: "team" 또는 "enterprise"):
     * price_per_user_per_month: 사용자당 월 가격 (USD) - 필수!
     * price_per_month: null 또는 전체 팀 월 가격 (USD)
   - 플랜 타입 (plan_type): "individual" (개인용), "team" (팀용), "enterprise" (엔터프라이즈)
   - 출처 URL (source_url): 가격 정보 출처
   - 🚨 **예시**:
     * 개인용: "Pro 플랜: $10/월" → plan_type: "individual", price_per_month: 10
     * 팀용: "Team 플랜: 사용자당 $19/월" → plan_type: "team", price_per_user_per_month: 19
     * 엔터프라이즈: "Enterprise: 사용자당 $25/월" → plan_type: "enterprise", price_per_user_per_month: 25
3. **통합 기능** (integrations): ⚠️ 예시: GitHub, GitLab, Slack, Jira 등은 참고용일 뿐, 실제 통합 서비스는 Findings에서 확인한 값 사용
4. **지원 언어** (supported_languages): ⚠️ 예시: Python, JavaScript, Java 등은 참고용일 뿐, 실제 지원 언어는 Findings에서 확인한 값 사용
   - 🚨 **중요**: 프레임워크나 런타임이 언급되면 해당 언어도 포함하세요!
     * 예: "Node.js 지원" → JavaScript 포함
     * 예: "React 지원" → JavaScript, TypeScript 포함
     * 예: "Spring Boot 지원" → Java 포함
     * 이는 일반적인 기술 지식이므로 Findings에 명시되지 않아도 포함하세요!
5. **보안 정책** (security_policy): "opt-in", "opt-out", "on-premise", "no-transmission"
6. **보안 상세** (security_details): 보안 정책 설명
7. **업무 지원** (workflow_support): "code_completion", "code_generation", "code_review", "refactoring", "debugging"
   - 🚨 **중요**: Findings에서 "PR 리뷰", "Pull Request 리뷰", "코드 리뷰", "PR 분석", "PR 자동화", "PR 코멘트" 같은 표현이 있으면 반드시 "code_review"를 포함하세요!
   - Findings에서 "자동완성", "코드 완성", "코드 제안" 같은 표현이 있으면 "code_completion"을 포함하세요!
   - Findings에서 "코드 생성", "파일 생성", "함수 생성" 같은 표현이 있으면 "code_generation"을 포함하세요!
8. **주요 기능** (primary_features)
9. **기능 카테고리** (feature_category): "code_completion", "code_review", "security_scan" 등
10. **출처 URL** (source_urls)

**중요:**
- Findings에 없는 정보는 추측하지 마세요!
- 가격은 반드시 Findings에서 확인한 실제 가격만 사용하세요!
- 플랜명은 도구마다 다를 수 있으므로 Findings에서 확인한 실제 플랜명을 사용하세요!

**출력 형식:**
JSON 배열로 반환하세요. 각 도구마다 하나의 객체:

```json
[
  {{
    "name": "도구명",
    "pricing_plans": [
      {{
        "name": "플랜명",
        "price_per_user_per_month": 숫자 또는 null,
        "price_per_month": 숫자 또는 null,
        "plan_type": "individual" | "team" | "enterprise",
        "source_url": "URL"
      }}
    ],
    "integrations": ["실제 통합 서비스명1", "실제 통합 서비스명2"],  # ⚠️ 위는 예시일 뿐, Findings에서 확인한 실제 값 사용
    "supported_languages": ["실제 지원 언어1", "실제 지원 언어2"],  # ⚠️ 위는 예시일 뿐, Findings에서 확인한 실제 값 사용
    "security_policy": "opt-in" | "opt-out" | "on-premise" | "no-transmission" | null,
    "security_details": "상세 설명",
    "workflow_support": ["code_completion", "code_generation"],
    "primary_features": ["기능1", "기능2"],
    "feature_category": "code_completion",
    "source_urls": ["URL1", "URL2"]
  }}
]
```

연구 결과:
{findings}

JSON 형식으로만 응답하세요 (설명 없이):
"""


async def extract_tool_facts(
    findings: str,
    config: RunnableConfig
) -> List[ToolFact]:
    """Findings에서 도구 사실 추출"""
    
    configurable = Configuration.from_runnable_config(config)
    
    model = init_chat_model(
        "gpt-4o-mini",
        config={
            "model": configurable.research_model,
            "max_tokens": 4096,
            "api_key": get_api_key_for_model(configurable.research_model, config),
        }
    )
    
    prompt = fact_extraction_prompt.format(findings=findings)
    
    try:
        response = await model.ainvoke([
            SystemMessage(content="당신은 연구 결과에서 구조화된 사실을 추출하는 전문가입니다. JSON 형식으로만 응답하세요."),
            HumanMessage(content=prompt)
        ])
        
        content = str(response.content).strip()
        
        # JSON 추출 (마크다운 코드 블록 제거)
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # 코드 블록 없이 JSON만 있는 경우
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
        
        facts_data = json.loads(content)
        
        # ToolFact 객체로 변환
        tool_facts = []
        for fact_data in facts_data:
            # pricing_plans 변환
            pricing_plans = []
            for plan_data in fact_data.get("pricing_plans", []):
                pricing_plans.append(PricingPlan(**plan_data))
            
            # security_policy 변환
            security_policy = None
            if fact_data.get("security_policy"):
                try:
                    security_policy = SecurityPolicy(fact_data["security_policy"])
                except ValueError:
                    security_policy = None
            
            # workflow_support 변환
            workflow_support = []
            for workflow in fact_data.get("workflow_support", []):
                try:
                    workflow_support.append(WorkflowType(workflow))
                except ValueError:
                    pass
            
            tool_fact = ToolFact(
                name=fact_data["name"],
                pricing_plans=pricing_plans,
                integrations=fact_data.get("integrations", []),
                supported_languages=fact_data.get("supported_languages", []),
                security_policy=security_policy,
                security_details=fact_data.get("security_details"),
                workflow_support=workflow_support,
                primary_features=fact_data.get("primary_features", []),
                feature_category=fact_data.get("feature_category", "code_completion"),
                source_urls=fact_data.get("source_urls", [])
            )
            
            tool_facts.append(tool_fact)
        
        return tool_facts
    
    except Exception as e:
        print(f"⚠️ [Fact Extractor] 오류: {e}")
        return []

