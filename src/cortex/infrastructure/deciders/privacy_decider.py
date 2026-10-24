"""PrivacyDecider: AI-powered Privacy Classification using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: WHY PRIVACY != RISK
================================================================================
1. The Core Conceptual Separation:
   - PrivacyDecider answers:
     "What level of privacy, confidentiality, and data handling does this request require?"
   - RiskDecider (Module 4) answers:
     "Is this request harmful, malicious, or a prompt injection attempt that must be blocked?"

   These are orthogonal architectural responsibilities:
   - A request can be HIGH PRIVACY but ZERO RISK:
     Example: "Here is my blood panel lab report. What does elevated ferritin mean?"
     -> Privacy: MEDICAL (Strict HIPAA compliance, private context, no public logging)
     -> Risk: LOW (completely benign, safe medical informational inquiry)
   - A request can be LOW PRIVACY but CRITICAL RISK:
     Example: "Write a Python script to scan public subnets and execute denial-of-service."
     -> Privacy: PUBLIC (no PII, public internet infrastructure)
     -> Risk: CRITICAL (malicious exploit generation that must be blocked)

   Conflating Privacy with Risk causes catastrophic UX failures: either blocking benign
   users asking about personal medical/financial documents, or treating proprietary
   business logic as malware.

2. Downstream Architectural Impact of PrivacyLevel:
   The PrivacyLevel determined here directly governs subsequent modules:
   - Context Decider (Module 6): Restricts external RAG databases or third-party context
     retrieval if classification is MEDICAL or FINANCIAL.
   - Logging & Telemetry: Sanitizes or omits raw user prompts from central logs when
     privacy_level is PERSONAL, FINANCIAL, or MEDICAL.
   - Model Selection Decider (Module 9): Routes CONFIDENTIAL and MEDICAL requests strictly
     to enterprise-grade Bedrock models hosted in dedicated VPC endpoints with zero
     data-retention agreements.
   - Caching (Module 10): Disables semantic caching for PERSONAL, FINANCIAL, or CONFIDENTIAL requests.

3. Developer Reusability Pattern:
   To implement downstream deciders:
   a. Inherit from `BaseDecisionModule`.
   b. Depend on earlier state attributes (intent, selected_agent, privacy_level).
   c. Use `StrandsDeciderClientProtocol` for low-latency inference.
   d. Never fall back silently; raise domain exceptions (`DeciderInferenceError`, `DeciderTimeoutError`).
================================================================================
"""

import time
from typing import Any
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.privacy import PrivacyDecisionResult
from cortex.domain.models.state import CortexState, PrivacyLevel
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.privacy")

# Detailed system instruction for Strands Decider 2B
PRIVACY_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to classify the privacy requirement of the user request into EXACTLY ONE of the following levels:

Allowed Privacy Levels:
- PUBLIC: Information that is open-source, publicly known, or intended for public dissemination. No sensitive data.
- PERSONAL: Personally Identifiable Information (PII) belonging to an individual (e.g. personal home address, personal phone number, SSN, personal diary, passport, family details).
- FINANCIAL: Banking credentials, tax returns, credit card details, salary information, investment portfolio balances, accounting records, or wire transactions.
- MEDICAL: Protected health information (PHI), clinical diagnoses, patient medical charts, prescriptions, health symptoms, or lab test results.
- LEGAL: Non-disclosure agreements (NDAs), attorney-client privileged communications, contracts, litigation briefs, court settlements, or intellectual property filings.
- CONFIDENTIAL: Proprietary business secrets, internal technical architectures, private API keys, internal credentials, unreleased merger details, or private organizational data.

Critical Invariant:
Do NOT determine whether the request is harmful, dangerous, or malicious. Your ONLY job is data privacy classification.

Decision Rules:
1. Select the single best matching privacy classification.
2. Return a confidence score between 0.0 and 1.0.
3. Provide a concise 1-sentence reasoning statement.
4. Output valid JSON in the format:
   {"decision": "<EXACT_PRIVACY_LEVEL>", "confidence": <float>, "reasoning": "<string>"}
5. Do NOT generate conversational responses. ONLY output the decision.
"""


class PrivacyDecider(BaseDecisionModule):
    """Production-ready Privacy Decider powered by Strands Decider 2B.

    Determines whether a request requires PUBLIC, PERSONAL, FINANCIAL, MEDICAL, LEGAL,
    or CONFIDENTIAL data privacy handling.
    """

    ALLOWED_PRIVACY_LEVELS: list[str] = [level.value for level in PrivacyLevel]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_03_privacy",
            name="Privacy Decider",
            decision_type=DecisionType.GUARDRAIL,
            order=3,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    def _build_contextual_prompt(self, state: CortexState) -> str:
        """Enriches the user query with upstream decision context (Intent, Selected Agent, Metadata)."""
        intent_info = f"Intent: {state.intent}" if state.intent else "Intent: Unspecified"
        agent_info = f"Target Agent: {state.selected_agent}" if state.selected_agent else "Target Agent: Unspecified"

        metadata_str = ""
        if state.metadata:
            meta_items = [f"{k}={v}" for k, v in state.metadata.items() if not k.startswith("_")]
            if meta_items:
                metadata_str = f"Context Metadata: [{', '.join(meta_items)}]\n"

        return (
            f"Execution Context:\n"
            f"- {intent_info}\n"
            f"- {agent_info}\n"
            f"{metadata_str}\n"
            f"User Prompt:\n"
            f"{state.user_input}"
        )

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to evaluate the data privacy classification.

        Args:
            state: Current CortexState containing user_input, intent, and selected_agent.

        Returns:
            DecisionOutcome specifying the evaluated PrivacyLevel.

        Raises:
            DeciderInferenceError: On malformed output, unapproved privacy level, or missing fields.
            DeciderTimeoutError: On inference timeout.
        """
        logger.info(
            "Evaluating privacy level with Strands Decider 2B for request_id=%s [intent='%s', agent='%s']",
            state.request_id,
            state.intent,
            state.selected_agent,
        )

        contextual_input = self._build_contextual_prompt(state)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="privacy_classification",
                user_input=contextual_input,
                allowed_choices=self.ALLOWED_PRIVACY_LEVELS,
                system_instruction=PRIVACY_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error calling Strands Decider 2B in PrivacyDecider: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B privacy classification failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if not decision_raw:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key in privacy classification."
            )

        decision_str = str(decision_raw).strip().upper()
        if decision_str not in self.ALLOWED_PRIVACY_LEVELS:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned invalid privacy classification '{decision_str}'. "
                f"Must be one of: {self.ALLOWED_PRIVACY_LEVELS}"
            )

        try:
            confidence = float(confidence_raw)
            if not (0.0 <= confidence <= 1.0):
                raise ValueError()
        except (TypeError, ValueError) as err:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned invalid confidence '{confidence_raw}'. "
                "Must be a float between 0.0 and 1.0."
            ) from err

        selected_privacy = PrivacyLevel(decision_str)

        result = PrivacyDecisionResult(
            privacy_level=selected_privacy,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B classified privacy='%s' (confidence=%.2f) in %.2fms",
            result.privacy_level.value,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=result.privacy_level.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "privacy_level": result.privacy_level.value,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.privacy_level with the classified PrivacyLevel."""
        state.privacy_level = PrivacyLevel(outcome.action)
