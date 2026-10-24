"""Factory registry providing the ordered list of 11 Strands Decider 2B modules."""

from cortex.domain.protocols.decider import DecisionModuleProtocol
from cortex.infrastructure.deciders.agent_decider import AgentSelectionDecider
from cortex.infrastructure.deciders.intent_decider import IntentDecider
from cortex.infrastructure.deciders.placeholders import (
    ComplexityDecider,
    ConfidenceDecider,
    ContextDecider,
    CostStrategyDecider,
    ModelSelectionDecider,
    ToolSelectionDecider,
    VisionRequirementDecider,
)
from cortex.infrastructure.deciders.privacy_decider import PrivacyDecider
from cortex.infrastructure.deciders.risk_decider import RiskDecider


def get_default_decision_modules(
    default_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
) -> list[DecisionModuleProtocol]:
    """Returns the complete, ordered pipeline of the 11 decision modules.

    Modules 1-4 (IntentDecider, AgentSelectionDecider, PrivacyDecider, RiskDecider) are production
    AI-powered modules backed by Strands Decider 2B.
    Modules 5-11 are placeholders ready for implementation.
    """
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
