"""Factory registry providing the ordered list of 11 Strands Decider 2B modules."""

from cortex.domain.protocols.decider import DecisionModuleProtocol
from cortex.infrastructure.deciders.placeholders import (
    AgentSelectionDecider,
    ComplexityDecider,
    ConfidenceDecider,
    ContextDecider,
    CostStrategyDecider,
    IntentDecider,
    ModelSelectionDecider,
    PrivacyDecider,
    RiskDecider,
    ToolSelectionDecider,
    VisionRequirementDecider,
)


def get_default_decision_modules(
    default_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> list[DecisionModuleProtocol]:
    """Returns the complete, ordered pipeline of the 11 decision modules."""
    modules: list[DecisionModuleProtocol] = [
        IntentDecider(),
        AgentSelectionDecider(),
        PrivacyDecider(),
        RiskDecider(),
        ToolSelectionDecider(),
        ContextDecider(),
        VisionRequirementDecider(),
        ComplexityDecider(),
        ModelSelectionDecider(default_model_id=default_model_id),
        CostStrategyDecider(),
        ConfidenceDecider(),
    ]
    # Ensure modules are strictly sorted by execution order
    return sorted(modules, key=lambda m: m.order)
