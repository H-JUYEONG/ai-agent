"""Decision Engine for tool recommendation"""

from typing import List, Dict, Optional
from app.agent.models import (
    ToolFact,
    UserContext,
    ToolScore,
    DecisionResult,
    SecurityPolicy,
    WorkflowType,
)


class DecisionEngine:
    """도구 추천 판단 엔진"""
    
    def __init__(
        self, 
        user_context: UserContext,
        weights: Optional[Dict[str, float]] = None
    ):
        self.user_context = user_context
        # 가중치 설정 (기본값, 필요시 변경 가능)
        self.weights = weights or {
            "language_support": 0.3,
            "integration": 0.2,
            "workflow_fit": 0.2,
            "price": 0.15,
            "security": 0.15
        }
    
    def filter_tools(self, tools: List[ToolFact]) -> List[ToolFact]:
        """하드 제약 조건으로 필터링"""
        filtered = []
        
        for tool in tools:
            # 제외 목록 확인
            if tool.name in self.user_context.excluded_tools:
                continue
            
            # 보안 요구사항 확인
            if self.user_context.security_required:
                if tool.security_policy not in [
                    SecurityPolicy.ON_PREMISE,
                    SecurityPolicy.NO_TRANSMISSION
                ]:
                    continue
            
            # 언어 지원 확인 (완화: 도구의 supported_languages가 비어있거나 정보가 없으면 필터링에서 제외하지 않음)
            # 대신 점수 계산 단계에서 처리 (language_support_score로 반영)
            if self.user_context.tech_stack:
                if not tool.supported_languages:
                    # supported_languages 정보가 없으면 필터링에서 제외하지 않음 (점수 계산에서 처리)
                    print(f"  ⚠️ [Filter] {tool.name}: supported_languages 정보 없음 → 필터링 통과 (점수 계산에서 처리)")
                else:
                    required_languages = [
                        lang.lower() for lang in self.user_context.tech_stack
                    ]
                    tool_languages = [
                        lang.lower() for lang in tool.supported_languages
                    ]
                    # 필수 언어 중 하나라도 지원하는지 확인
                    has_match = any(
                        req_lang in tool_lang or tool_lang in req_lang
                        for req_lang in required_languages
                        for tool_lang in tool_languages
                    )
                    if not has_match:
                        # 필수 언어가 하나라도 지원되지 않으면 필터링에서 제외하지 않음 (점수 계산에서 처리)
                        print(f"  ⚠️ [Filter] {tool.name}: 필수 언어({required_languages}) 미지원, 지원 언어({tool_languages}) → 필터링 통과 (점수 계산에서 감점)")
                    else:
                        print(f"  ✅ [Filter] {tool.name}: 필수 언어 일부 지원 확인")
                # 필터링 단계에서는 제외하지 않음, 점수 계산에서 반영
            
            # 통합 기능 확인
            if self.user_context.required_integrations:
                tool_integrations = [
                    integ.lower() for integ in tool.integrations
                ]
                required_integrations = [
                    integ.lower() for integ in self.user_context.required_integrations
                ]
                if not any(
                    req_integ in tool_integ or tool_integ in req_integ
                    for req_integ in required_integrations
                    for tool_integ in tool_integrations
                ):
                    # 필수 통합이 하나도 없으면 제외
                    continue
            
            # 🆕 업무 요구사항 필터링 (완화: 코드 리뷰가 필수여도 다른 기능을 지원하면 통과)
            if self.user_context.workflow_focus:
                # 모든 필수 업무를 지원하지 않아도, 하나라도 지원하면 통과 (점수 계산에서 감점)
                # 완전히 제외하는 대신, 점수 계산 단계에서 부분 점수를 주는 것이 더 나음
                # 예: code_review 필수인데 지원 안 하면 workflow_fit_score가 낮아지지만 완전히 제외하지는 않음
                pass  # 필터링 단계에서는 제외하지 않음, 점수 계산에서 반영
            
            filtered.append(tool)
        
        return filtered
    
    def calculate_score(self, tool: ToolFact) -> ToolScore:
        """도구 점수 계산"""
        exclusion_reason = None
        
        # 1. 언어 지원 점수 (완화: 정보가 부족하거나 부분 지원해도 점수 부여)
        language_support_score = 1.0  # 기본값: 만점 (정보가 없으면 관대하게)
        if self.user_context.tech_stack:
            required_languages = [
                lang.lower() for lang in self.user_context.tech_stack
            ]
            if tool.supported_languages:
                tool_languages = [
                    lang.lower() for lang in tool.supported_languages
                ]
                matches = sum(
                    1 for req_lang in required_languages
                    for tool_lang in tool_languages
                    if req_lang in tool_lang or tool_lang in req_lang
                )
                if required_languages:
                    # 일부만 지원해도 부분 점수 부여 (0.5 이상 보장)
                    language_support_score = max(0.5, matches / len(required_languages))
            else:
                # tool.supported_languages 정보가 없으면 중간 점수 (완전 제외 방지)
                language_support_score = 0.7  # 정보 부족해도 관대하게
        
        # 2. 통합 기능 점수 (완화: 정보 부족해도 부분 점수 부여)
        integration_score = 0.7  # 기본값: 높은 점수 (정보가 없으면 관대하게)
        if self.user_context.required_integrations:
            if tool.integrations:
                tool_integrations = [
                    integ.lower() for integ in tool.integrations
                ]
                required_integrations = [
                    integ.lower() for integ in self.user_context.required_integrations
                ]
                matches = sum(
                    1 for req_integ in required_integrations
                    for tool_integ in tool_integrations
                    if req_integ in tool_integ or tool_integ in req_integ
                )
                if required_integrations:
                    # 일부만 지원해도 부분 점수 부여 (최소 0.5 보장)
                    integration_score = max(0.5, matches / len(required_integrations))
            else:
                # tool.integrations 정보가 없으면 중간 점수 (완전 제외 방지)
                integration_score = 0.6  # 정보 부족해도 관대하게
        
        # 3. 업무 적합성 점수 (완화: 정보 부족해도 부분 점수 부여)
        workflow_fit_score = 0.8  # 기본값: 높은 점수 (정보가 없으면 관대하게)
        if self.user_context.workflow_focus:
            if tool.workflow_support:
                matches = sum(
                    1 for workflow in self.user_context.workflow_focus
                    if workflow in tool.workflow_support
                )
                # 필수 요구사항이 모두 지원되면 만점, 일부만 지원되면 부분 점수 (최소 0.4 보장)
                if self.user_context.workflow_focus:
                    workflow_fit_score = max(0.4, matches / len(self.user_context.workflow_focus))
                
                # 코드 리뷰가 요청되었지만 지원하지 않으면 감점 (완전 제외는 하지 않음)
                if WorkflowType.CODE_REVIEW in self.user_context.workflow_focus:
                    if WorkflowType.CODE_REVIEW not in tool.workflow_support:
                        # 코드 작성 기능이 있으면 최소 0.4점 부여 (완전 제외 방지)
                        if WorkflowType.CODE_COMPLETION in tool.workflow_support or WorkflowType.CODE_GENERATION in tool.workflow_support:
                            workflow_fit_score = max(0.4, workflow_fit_score * 0.6)  # 최소 0.4점, 최대 40% 감점
                        else:
                            # 코드 작성 기능도 없으면 더 낮은 점수 (하지만 완전 0점은 아님)
                            workflow_fit_score = max(0.2, workflow_fit_score * 0.3)
            else:
                # tool.workflow_support 정보가 없으면 중간 점수 (완전 제외 방지)
                workflow_fit_score = 0.6  # 정보 부족해도 관대하게
        
        # 4. 가격 점수 (예산이 없어도 상대적 비교)
        price_score = 1.0
        tool_monthly_cost = None
        
        if self.user_context.team_size:
            # 팀용 플랜 찾기
            team_plans = [
                plan for plan in tool.pricing_plans
                if plan.plan_type in ["team", "business", "enterprise"]
            ]
            if team_plans:
                cheapest_plan = min(
                    team_plans,
                    key=lambda p: p.price_per_user_per_month or float('inf')
                )
                if cheapest_plan.price_per_user_per_month:
                    tool_monthly_cost = cheapest_plan.price_per_user_per_month * self.user_context.team_size
        else:
            # 개인용 플랜 찾기
            individual_plans = [
                plan for plan in tool.pricing_plans
                if plan.plan_type == "individual"
            ]
            if individual_plans:
                cheapest_plan = min(
                    individual_plans,
                    key=lambda p: p.price_per_month or float('inf')
                )
                if cheapest_plan.price_per_month:
                    tool_monthly_cost = cheapest_plan.price_per_month
        
        if tool_monthly_cost is not None:
            if self.user_context.budget_max:
                # 예산이 있으면 예산 기준으로 점수 계산
                if tool_monthly_cost > self.user_context.budget_max:
                    price_score = 0.0
                    exclusion_reason = f"예산 초과: ${tool_monthly_cost:.2f}/월 > ${self.user_context.budget_max:.2f}/월"
                else:
                    budget_ratio = tool_monthly_cost / self.user_context.budget_max
                    price_score = max(0.0, 1.0 - (budget_ratio - 0.5) * 0.5) if budget_ratio > 0.5 else 1.0
            else:
                # 예산이 없으면 가격 정보만 저장 (나중에 상대적 비교용)
                # 가격 점수는 기본값 1.0 유지 (상대적 비교는 make_decision에서)
                pass
        
        # 5. 보안 점수
        security_score = 1.0
        if self.user_context.security_required:
            if tool.security_policy in [
                SecurityPolicy.ON_PREMISE,
                SecurityPolicy.NO_TRANSMISSION
            ]:
                security_score = 1.0
            elif tool.security_policy == SecurityPolicy.OPT_OUT:
                security_score = 0.5  # 선택적으로 차단 가능
            else:
                security_score = 0.0
                if not exclusion_reason:
                    exclusion_reason = "보안 요구사항 미충족"
        
        # 총점 계산 (가중 평균) - 가중치는 설정 가능
        total_score = (
            language_support_score * self.weights["language_support"] +
            integration_score * self.weights["integration"] +
            workflow_fit_score * self.weights["workflow_fit"] +
            price_score * self.weights["price"] +
            security_score * self.weights["security"]
        )
        
        return ToolScore(
            tool_name=tool.name,
            total_score=total_score,
            language_support_score=language_support_score,
            integration_score=integration_score,
            workflow_fit_score=workflow_fit_score,
            price_score=price_score,
            security_score=security_score,
            exclusion_reason=exclusion_reason
        )
    
    def remove_duplicate_features(self, tools: List[ToolFact], scores: List[ToolScore]) -> List[ToolFact]:
        """같은 기능을 하는 도구 중 점수가 높은 것만 남기기"""
        # 기능 카테고리별로 그룹화
        category_groups: Dict[str, List[tuple[ToolFact, ToolScore]]] = {}
        
        for tool, score in zip(tools, scores):
            category = tool.feature_category or "code_completion"  # 기본값 설정
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append((tool, score))
        
        # 각 카테고리에서 점수가 가장 높은 것만 선택
        selected_tools = []
        print(f"🔍 [Duplicate Removal] 카테고리 그룹: {list(category_groups.keys())}")
        for category, tool_score_pairs in category_groups.items():
            # 점수 순으로 정렬
            tool_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
            print(f"🔍 [Duplicate Removal] {category} 카테고리: {[(t.name, s.total_score) for t, s in tool_score_pairs]}")
            # 가장 높은 점수만 선택 (같은 점수면 첫 번째만)
            selected_tool = tool_score_pairs[0][0]
            selected_score = tool_score_pairs[0][1]
            selected_tools.append(selected_tool)
            print(f"✅ [Duplicate Removal] {category} 카테고리에서 {selected_tool.name} 선택 (점수: {selected_score.total_score:.3f})")
        
        return selected_tools
    
    def make_decision(self, tools: List[ToolFact]) -> DecisionResult:
        """최종 판단"""
        # 🚨 디버깅: 필터링 전 도구 목록
        print("=" * 80)
        print("🔍 [Decision Engine] 필터링 전")
        print(f"  입력 도구 개수: {len(tools)}개")
        print(f"  입력 도구명: {[tool.name for tool in tools[:10]]}")
        print("=" * 80)
        
        # 1. 필터링
        filtered_tools = self.filter_tools(tools)
        
        # 🚨 디버깅: 필터링 후 도구 목록
        print("=" * 80)
        print("🔍 [Decision Engine] 필터링 후")
        print(f"  필터링 후 도구 개수: {len(filtered_tools)}개")
        if filtered_tools:
            print(f"  필터링 후 도구명: {[tool.name for tool in filtered_tools]}")
        else:
            print("  ⚠️ 필터링 후 도구가 없습니다!")
            print(f"  필터링 조건:")
            print(f"    - 제외 목록: {self.user_context.excluded_tools}")
            print(f"    - 보안 요구: {self.user_context.security_required}")
            print(f"    - 필수 언어: {self.user_context.tech_stack}")
            print(f"    - 필수 통합: {self.user_context.required_integrations}")
            print(f"    - 필수 업무: {[w.value for w in self.user_context.workflow_focus]}")
        print("=" * 80)
        
        # 2. 점수 계산
        if not filtered_tools:
            # 필터링 후 도구가 없으면 빈 결과 반환
            print("⚠️ [Decision Engine] 필터링 후 도구가 없어 Decision Engine 실행 불가")
            return DecisionResult(
                recommended_tools=[],
                excluded_tools=[tool.name for tool in tools],
                tool_scores=[],
                reasoning={}
            )
        
        scores = [self.calculate_score(tool) for tool in filtered_tools]
        
        # 🚨 디버깅: 스코어링 결과
        print("=" * 80)
        print("🔍 [Decision Engine] 스코어링 결과")
        if scores:
            for score in scores[:5]:  # 상위 5개만 출력
                print(f"  {score.tool_name}:")
                print(f"    총점: {score.total_score:.3f}")
                print(f"    언어 지원: {score.language_support_score:.3f}")
                print(f"    통합: {score.integration_score:.3f}")
                print(f"    업무 적합성: {score.workflow_fit_score:.3f}")
                print(f"    가격: {score.price_score:.3f}")
                print(f"    보안: {score.security_score:.3f}")
                if score.exclusion_reason:
                    print(f"    제외 이유: {score.exclusion_reason}")
        else:
            print("  ⚠️ 스코어링 결과가 없습니다!")
        print("=" * 80)
        
        # 🆕 2-1. 예산이 없으면 가격 상대적 비교로 점수 조정
        if not self.user_context.budget_max and self.user_context.team_size:
            # 모든 도구의 월간 비용 계산
            tool_costs = {}
            for tool, score in zip(filtered_tools, scores):
                team_plans = [p for p in tool.pricing_plans if p.plan_type in ["team", "business", "enterprise"]]
                if team_plans:
                    cheapest = min(team_plans, key=lambda p: p.price_per_user_per_month or float('inf'))
                    if cheapest.price_per_user_per_month:
                        tool_costs[tool.name] = cheapest.price_per_user_per_month * self.user_context.team_size
            
            if tool_costs:
                # 가장 저렴한 도구를 기준으로 가격 점수 조정
                min_cost = min(tool_costs.values())
                max_cost = max(tool_costs.values())
                cost_range = max_cost - min_cost if max_cost > min_cost else 1.0
                
                for tool, score in zip(filtered_tools, scores):
                    if tool.name in tool_costs:
                        # 비용이 낮을수록 높은 점수 (0.5 ~ 1.0 범위)
                        normalized_cost = (tool_costs[tool.name] - min_cost) / cost_range if cost_range > 0 else 0
                        price_score = 1.0 - (normalized_cost * 0.5)  # 최저가면 1.0, 최고가면 0.5
                        # 기존 가격 점수와 평균 (다른 요소도 고려)
                        score.price_score = (score.price_score + price_score) / 2
                        # 총점 재계산
                        score.total_score = (
                            score.language_support_score * self.weights["language_support"] +
                            score.integration_score * self.weights["integration"] +
                            score.workflow_fit_score * self.weights["workflow_fit"] +
                            score.price_score * self.weights["price"] +
                            score.security_score * self.weights["security"]
                        )
        
        # 3. 제외된 도구 찾기
        excluded_tools = [
            score.tool_name
            for score in scores
            if score.total_score == 0.0 or score.exclusion_reason
        ]
        
        # 4. 점수가 0인 도구 제외
        valid_tools = [
            tool for tool, score in zip(filtered_tools, scores)
            if score.total_score > 0.0 and not score.exclusion_reason
        ]
        valid_scores = [
            score for score in scores
            if score.total_score > 0.0 and not score.exclusion_reason
        ]
        
        # 5. 중복 기능 제거
        if len(valid_tools) > 1:
            unique_tools = self.remove_duplicate_features(valid_tools, valid_scores)
            # 제거된 도구를 excluded에 추가
            removed_tools = [
                tool.name for tool in valid_tools
                if tool not in unique_tools
            ]
            excluded_tools.extend(removed_tools)
            valid_tools = unique_tools
            # 점수도 다시 계산
            valid_scores = [
                score for score in valid_scores
                if score.tool_name in [tool.name for tool in valid_tools]
            ]
        
        # 6. 점수 순으로 정렬
        tool_score_pairs = list(zip(valid_tools, valid_scores))
        tool_score_pairs.sort(key=lambda x: x[1].total_score, reverse=True)
        
        recommended_tools = [tool.name for tool, _ in tool_score_pairs]
        
        # 7. 판단 이유 생성 (더 구체적으로)
        reasoning = {}
        for tool, score in tool_score_pairs:
            reasons = []
            
            # 언어 지원
            if score.language_support_score == 1.0:
                if self.user_context.tech_stack:
                    reasons.append(f"기술 스택({', '.join(self.user_context.tech_stack)}) 완벽 지원")
                else:
                    reasons.append("기술 스택 완벽 지원")
            elif score.language_support_score > 0.5:
                reasons.append(f"기술 스택({', '.join(self.user_context.tech_stack)}) 부분 지원 ({score.language_support_score:.0%})")
            elif score.language_support_score > 0:
                reasons.append(f"기술 스택 지원 부족 ({score.language_support_score:.0%})")
            
            # 업무 적합성
            if score.workflow_fit_score == 1.0:
                workflow_names = [w.value for w in self.user_context.workflow_focus] if self.user_context.workflow_focus else []
                if workflow_names:
                    reasons.append(f"필수 업무({', '.join(workflow_names)}) 완벽 지원")
                else:
                    reasons.append("업무 적합성 높음")
            elif score.workflow_fit_score > 0.5:
                reasons.append(f"업무 적합성 부분 지원 ({score.workflow_fit_score:.0%})")
            
            # 통합 기능
            if score.integration_score == 1.0:
                reasons.append("필수 통합 기능 완벽 지원")
            elif score.integration_score > 0.5:
                reasons.append("통합 기능 부분 지원")
            
            # 가격
            if self.user_context.team_size:
                team_plans = [p for p in tool.pricing_plans if p.plan_type in ["team", "business", "enterprise"]]
                if team_plans:
                    cheapest = min(team_plans, key=lambda p: p.price_per_user_per_month or float('inf'))
                    if cheapest.price_per_user_per_month:
                        monthly = cheapest.price_per_user_per_month * self.user_context.team_size
                        annual = monthly * 12
                        reasons.append(f"비용 효율적 (${monthly:.0f}/월, ${annual:.0f}/년)")
            
            if score.price_score == 1.0 and self.user_context.budget_max:
                reasons.append("예산 범위 내")
            
            # 보안
            if score.security_score == 1.0 and self.user_context.security_required:
                reasons.append("보안 요구사항 충족")
            
            reasoning[tool.name] = "; ".join(reasons) if reasons else "기본 요구사항 충족"
        
        return DecisionResult(
            recommended_tools=recommended_tools,
            excluded_tools=excluded_tools,
            tool_scores=[score for _, score in tool_score_pairs],
            reasoning=reasoning
        )

