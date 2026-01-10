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
   - **연간 플랜** (plan_type: "team" 또는 "enterprise"):
     * "연간", "년간", "per year", "annually", "$X/년", "$X/year" 같은 표현이 있으면 연간 플랜으로 인식
     * price_per_year: 전체 팀 연간 가격 (USD) - "연간 $4,200 (10명 기준)" 같은 경우
     * price_per_user_per_year: 사용자당 연간 가격 (USD) - "사용자당 연간 $500" 같은 경우
     * price_per_month: null
     * price_per_user_per_month: null
   - **사용량 기반 과금** (plan_type: "usage-based"):
     * "사용량 기반", "API 호출당", "토큰 기반", "입력/출력 토큰", "usage-based", "per API call" 같은 표현이 있으면 plan_type: "usage-based"
     * price_per_month: null
     * price_per_user_per_month: null
     * Findings에서 확인한 사용량 기반 가격 정보를 name 필드에 포함 (예: "입력: $1.50/백만 토큰, 출력: $6.00/백만 토큰")
   - 플랜 타입 (plan_type): "individual" (개인용), "team" (팀용), "enterprise" (엔터프라이즈), "usage-based" (사용량 기반)
   - 출처 URL (source_url): 가격 정보 출처
   - 🚨 **예시**:
     * 개인용: "Pro 플랜: $10/월" → plan_type: "individual", price_per_month: 10
     * 팀용: "Team 플랜: 사용자당 $19/월" → plan_type: "team", price_per_user_per_month: 19
     * 엔터프라이즈: "Enterprise: 사용자당 $25/월" → plan_type: "enterprise", price_per_user_per_month: 25
     * 연간 플랜: "연간 $4,200 (10명 기준)" → plan_type: "team", price_per_year: 4200
     * 연간 플랜: "사용자당 연간 $500" → plan_type: "team", price_per_user_per_year: 500
     * 사용량 기반: "입력: $1.50/백만 토큰, 출력: $6.00/백만 토큰" → plan_type: "usage-based", name: "입력: $1.50/백만 토큰, 출력: $6.00/백만 토큰"
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
   - 🚨🚨🚨 **매우 중요: 코드 리뷰 기능 추출**: Findings에서 "PR 리뷰", "Pull Request 리뷰", "코드 리뷰", "PR 분석", "PR 자동화", "PR 코멘트", "pull request review", "code review", "automated review", "review comments", "PR feedback" 같은 표현이 **하나라도** 있으면 반드시 "code_review"를 포함하세요!
   - Findings에서 "자동완성", "코드 완성", "코드 제안", "autocomplete", "code suggestion" 같은 표현이 있으면 "code_completion"을 포함하세요!
   - Findings에서 "코드 생성", "파일 생성", "함수 생성", "code generation", "file generation" 같은 표현이 있으면 "code_generation"을 포함하세요!
   - 🚨 **중요**: 사용자가 "코드 작성과 리뷰"를 요청했다면, Findings에서 리뷰 기능 관련 표현을 특히 주의 깊게 찾으세요!
8. **주요 기능** (primary_features)
9. **기능 카테고리** (feature_category): "code_completion", "code_review", "security_scan" 등
   - 🚨 **필수 필드**: 반드시 포함하세요! 도구의 주요 기능에 따라 분류하세요!
   - 코드 자동완성/생성 도구 → "code_completion"
   - PR/코드 리뷰 도구 → "code_review"
   - 보안 스캔 도구 → "security_scan"
   - 기본값은 "code_completion"이지만, 반드시 명시적으로 추출하세요!
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
        "price_per_year": 숫자 또는 null,
        "price_per_user_per_year": 숫자 또는 null,
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

**🚨 매우 중요:**
1. 최소한 1개 도구는 반드시 추출하세요! Findings에 도구 정보가 있으면 무조건 추출하세요!
2. 도구명(name)은 필수입니다! 없으면 추출할 수 없습니다!
3. 가격 정보가 없어도 괜찮습니다. 가격 플랜이 비어있어도 도구는 추출하세요!
4. 모든 필드가 완벽하지 않아도 괜찮습니다. 최소한 도구명만 있어도 추출하세요!
5. JSON 형식이 완벽하지 않아도 괜찮습니다. 최소한 유효한 JSON 배열 형태로만 만들어주세요!

JSON 형식으로만 응답하세요 (설명 없이, 바로 JSON 배열 시작):
"""


async def extract_tool_facts(
    findings: str,
    config: RunnableConfig,
    max_retries: int = 3
) -> List[ToolFact]:
    """Findings에서 도구 사실 추출 (재시도 로직 포함, 부분 성공 허용)"""
    
    configurable = Configuration.from_runnable_config(config)
    
    # 다른 노드들과 동일한 방식으로 모델 초기화
    from langchain.chat_models import init_chat_model
    configurable_model = init_chat_model(
        configurable_fields=("model", "max_tokens", "api_key"),
    )
    
    model_config = {
        "model": configurable.research_model,
        "max_tokens": 4096,
        "api_key": get_api_key_for_model(configurable.research_model, config),
    }
    
    model = configurable_model.with_config(model_config)
    
    prompt = fact_extraction_prompt.format(findings=findings)
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            print(f"🔍 [Fact Extractor] 추출 시도 {attempt + 1}/{max_retries}")
            
            response = await model.ainvoke([
                SystemMessage(content="당신은 연구 결과에서 구조화된 사실을 추출하는 전문가입니다. JSON 형식으로만 응답하세요. 최소한 도구명(name)은 반드시 포함하세요."),
                HumanMessage(content=prompt)
            ])
            
            content = str(response.content).strip()
            print(f"🔍 [Fact Extractor] LLM 응답 길이: {len(content)}자")
            print(f"🔍 [Fact Extractor] LLM 응답 시작 200자: {content[:200]}")
            
            # JSON 추출 (여러 패턴 시도)
            json_content = None
            
            # 패턴 1: ```json ... ``` 블록
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            
            # 패턴 2: ``` ... ``` 블록 (언어 지정 없음)
            if not json_content:
                json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    # JSON인지 확인
                    if not json_content.strip().startswith('[') and not json_content.strip().startswith('{'):
                        json_content = None
            
            # 패턴 3: [...] 배열 직접 찾기
            if not json_content:
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(0)
            
            # 패턴 4: 전체를 JSON으로 시도
            if not json_content:
                json_content = content.strip()
            
            if not json_content:
                print(f"⚠️ [Fact Extractor] JSON 내용을 찾을 수 없음")
                if attempt < max_retries - 1:
                    continue
                return []
            
            # JSON 파싱 시도
            try:
                facts_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"⚠️ [Fact Extractor] JSON 파싱 실패: {e}")
                print(f"🔍 [Fact Extractor] 파싱 시도한 내용: {json_content[:500]}")
                
                # JSON 복구 시도: 불완전한 JSON 마지막 부분 자르기
                if attempt < max_retries - 1:
                    # 마지막 불완전한 객체 제거 시도
                    json_content_fixed = json_content.rsplit('}', 1)[0] + '}]'
                    try:
                        facts_data = json.loads(json_content_fixed)
                        print(f"✅ [Fact Extractor] JSON 복구 성공")
                    except:
                        continue
                else:
                    return []
            
            if not isinstance(facts_data, list):
                print(f"⚠️ [Fact Extractor] JSON이 배열이 아님: {type(facts_data)}")
                if isinstance(facts_data, dict) and "tools" in facts_data:
                    facts_data = facts_data["tools"]
                elif isinstance(facts_data, dict) and "results" in facts_data:
                    facts_data = facts_data["results"]
                else:
                    facts_data = [facts_data]
            
            # ToolFact 객체로 변환 (부분 성공 허용)
            tool_facts = []
            success_count = 0
            fail_count = 0
            
            for idx, fact_data in enumerate(facts_data):
                try:
                    # 필수 필드 검증
                    if not fact_data.get("name"):
                        print(f"⚠️ [Fact Extractor] 도구 {idx+1}: name 필드 없음, 스킵")
                        fail_count += 1
                        continue
                    
                    # pricing_plans 변환 (부분 실패 허용)
                    pricing_plans = []
                    for plan_idx, plan_data in enumerate(fact_data.get("pricing_plans", [])):
                        try:
                            # 필수 필드 검증
                            if not plan_data.get("plan_type"):
                                # plan_type 추론 시도
                                plan_name = plan_data.get("name", "").lower()
                                plan_name_full = plan_name + " " + str(fact_data.get("name", "")).lower()
                                # 사용량 기반 확인 (가장 먼저)
                                if any(keyword in plan_name_full for keyword in ["usage-based", "usage based", "사용량 기반", "api 호출당", "api 호출", "토큰 기반", "입력/출력", "per api call", "per token", "per million tokens"]):
                                    plan_data["plan_type"] = "usage-based"
                                elif any(keyword in plan_name for keyword in ["team", "business", "enterprise"]):
                                    plan_data["plan_type"] = "team"
                                elif any(keyword in plan_name for keyword in ["individual", "personal", "pro"]):
                                    plan_data["plan_type"] = "individual"
                                else:
                                    plan_data["plan_type"] = "individual"  # 기본값
                                print(f"🔍 [Fact Extractor] plan_type 자동 추론: {plan_data['plan_type']}")
                            
                            pricing_plans.append(PricingPlan(**plan_data))
                        except Exception as e:
                            print(f"⚠️ [Fact Extractor] 플랜 {plan_idx+1} 변환 실패: {e}, 스킵")
                            continue
                    
                    # security_policy 변환
                    security_policy = None
                    if fact_data.get("security_policy"):
                        try:
                            security_policy = SecurityPolicy(fact_data["security_policy"])
                        except ValueError:
                            security_policy = None
                    
                    # tool_name 먼저 정의 (다른 곳에서 사용하므로)
                    tool_name = fact_data["name"]
                    
                    # workflow_support 변환
                    workflow_support = []
                    for workflow in fact_data.get("workflow_support", []):
                        try:
                            workflow_support.append(WorkflowType(workflow))
                        except ValueError:
                            pass
                    
                    # feature_category 기반으로 workflow_support 자동 추가
                    # Findings에서 명확히 추출하지 못했더라도 feature_category를 보고 추정
                    if not workflow_support:
                        feature_category = fact_data.get("feature_category", "code_completion")
                        tool_name_lower = tool_name.lower()
                        findings_text = fact_data.get("findings_text", "").lower()
                        
                        # 코드 리뷰 관련 키워드 확인
                        review_keywords = ["code review", "pr review", "pull request", "코드 리뷰", "pr 리뷰", "리뷰"]
                        if (feature_category == "code_review" or 
                            any(kw in tool_name_lower for kw in ["review", "리뷰", "codacy", "sonarqube", "qodo", "code-rabbit", "coderabbit", "greptile"]) or
                            any(kw in findings_text for kw in review_keywords)):
                            workflow_support.append(WorkflowType.CODE_REVIEW)
                        
                        # 코드 생성/완성 관련
                        if (feature_category == "code_completion" or 
                            "completion" in tool_name_lower or "autocomplete" in tool_name_lower):
                            workflow_support.append(WorkflowType.CODE_COMPLETION)
                        
                        if (feature_category == "code_generation" or 
                            "generation" in tool_name_lower or "generate" in tool_name_lower):
                            workflow_support.append(WorkflowType.CODE_GENERATION)
                        
                        # 기본값: code_completion
                        if not workflow_support:
                            workflow_support.append(WorkflowType.CODE_COMPLETION)
                    
                    # feature_category 기본값 설정 (None이면 기본값 사용)
                    feature_category = fact_data.get("feature_category") or "code_completion"
                    
                    # OpenAI Codex 특별 처리: Codex는 독립 제품이 아니라 API 기능이므로 적절히 처리
                    if "codex" in tool_name.lower() and "openai" in tool_name.lower():
                        # Codex는 code_generation 기능에 특화되어 있음
                        if not workflow_support or WorkflowType.CODE_GENERATION not in workflow_support:
                            workflow_support = [WorkflowType.CODE_GENERATION] + workflow_support
                        if feature_category != "code_review" and feature_category != "security_scan":
                            feature_category = "code_generation"
                    
                    tool_fact = ToolFact(
                        name=tool_name,
                        pricing_plans=pricing_plans,
                        integrations=fact_data.get("integrations", []),
                        supported_languages=fact_data.get("supported_languages", []),
                        security_policy=security_policy,
                        security_details=fact_data.get("security_details"),
                        workflow_support=workflow_support if workflow_support else [WorkflowType.CODE_COMPLETION],  # 기본값
                        primary_features=fact_data.get("primary_features", []),
                        feature_category=feature_category,
                        source_urls=fact_data.get("source_urls", [])
                    )
                    
                    tool_facts.append(tool_fact)
                    success_count += 1
                    print(f"✅ [Fact Extractor] 도구 {idx+1} 추출 성공: {fact_data['name']}")
                    
                except Exception as e:
                    print(f"⚠️ [Fact Extractor] 도구 {idx+1} 변환 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    fail_count += 1
                    continue
            
            # 부분 성공 허용: 최소 1개 도구라도 추출되면 성공
            if tool_facts:
                print(f"✅ [Fact Extractor] 추출 완료: {success_count}개 성공, {fail_count}개 실패")
                return tool_facts
            else:
                print(f"⚠️ [Fact Extractor] 모든 도구 추출 실패 ({fail_count}개 실패)")
                if attempt < max_retries - 1:
                    print(f"🔄 [Fact Extractor] 재시도 중...")
                    continue
                return []
        
        except Exception as e:
            print(f"⚠️ [Fact Extractor] 시도 {attempt + 1} 오류: {e}")
            import traceback
            traceback.print_exc()
            if attempt < max_retries - 1:
                print(f"🔄 [Fact Extractor] 재시도 중...")
                continue
            return []
    
    print(f"❌ [Fact Extractor] 최대 재시도 횟수 초과")
    return []

