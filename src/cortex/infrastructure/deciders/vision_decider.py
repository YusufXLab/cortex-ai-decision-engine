"""VisionRequirementDecider: Determines required multimodal and visual understanding capabilities using Strands Decider 2B.

================================================================================
ARCHITECTURE & DESIGN RATIONALE: TOOL SELECTION VS VISION REQUIREMENT DECIDER
================================================================================
1. The Core Conceptual Separation:
   - Tool Selection (Module 5) answers:
     "What external capabilities or I/O tools does the agent have access to?"
     (e.g., TOOL_SELECTION selects ["VISION", "FILE_READER"] if visual inspection capabilities are available).
   - VisionRequirementDecider (Module 7) answers:
     "What specific type of visual processing does this request actually require?"
     (e.g., OCR, IMAGE_UNDERSTANDING, DIAGRAM_ANALYSIS, UI_ANALYSIS, MULTIMODAL, or NONE).

   A tool is an execution capability flag; a vision requirement is a cognitive modality:
   - Even if "VISION" was selected as a permissible tool, VisionRequirementDecider determines
     whether visual reasoning is *actually* necessary for the user's input.
   - If the user provided a pure text prompt with no images or visual assets,
     VisionRequirementDecider returns "NONE", saving downstream multimodal processing overhead.
   - Furthermore, this module does NOT perform image understanding itself. Strands Decider 2B
     only decides the required vision capability category, preserving strict decider boundaries.

2. Impact on Downstream Model Routing & Execution (Module 9):
   - NONE:
     Enables Model Routing to select ultra-fast, cost-effective text-only models
     (e.g., Claude 3 Haiku, Amazon Titan Text, Meta Llama 3 8B), reducing latency by up to 70%
     and eliminating expensive vision token charges.
   - OCR:
     Enables the pipeline to route to specialized, high-accuracy document OCR engines
     (e.g., Amazon Textract or specialized OCR pipelines) to extract structured text before
     model ingestion, avoiding costly vision-encoder tokenization in frontier models.
   - IMAGE_UNDERSTANDING:
     Routes requests to vision-capable models (e.g., Bedrock Claude 3.5 Sonnet / Claude 3 Haiku)
     with general scene and aesthetic comprehension prompts.
   - DIAGRAM_ANALYSIS:
     Routes to frontier multimodal models equipped for high-resolution structural and spatial
     reasoning (flowcharts, AWS cloud architecture diagrams, UML models, entity-relationship schemas).
   - UI_ANALYSIS:
     Directs execution to visual models capable of bounding-box coordinates, component hierarchy,
     and software UI/UX comprehension (web screenshots, mobile app mockups, frontend bugs).
   - MULTIMODAL:
     Directs execution to large-context multimodal engines capable of interleaved cross-modal reasoning
     across mixed PDFs with tables, figures, embedded charts, and narrative text.
================================================================================
"""

import time
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.models.decision import DecisionOutcome, DecisionType
from cortex.domain.models.state import CortexState
from cortex.domain.models.vision import VisionDecisionResult, VisionRequirementType
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol
from cortex.infrastructure.deciders.base import BaseDecisionModule
from cortex.infrastructure.deciders.client import StrandsDeciderClient

logger = get_logger("cortex.deciders.vision")

VISION_REQUIREMENT_SYSTEM_INSTRUCTION = """You are Strands Decider 2B, an ultra-fast, deterministic decision engine.
Your sole job is to determine whether multimodal visual processing is required and, if so, what specific type of visual understanding is needed.

This module does NOT perform image understanding. You only decide the required vision capability.

Allowed Vision Requirements:
- NONE: No visual or multimodal processing required (e.g., text-only prompts, no uploaded images).
- OCR: Optical Character Recognition for text extraction from receipts, invoices, scanned forms, documents.
- IMAGE_UNDERSTANDING: General photo, scene, object, or artistic image description and inspection.
- DIAGRAM_ANALYSIS: Technical architecture diagrams, cloud topology, flowcharts, UML, sequence diagrams, system schemas.
- UI_ANALYSIS: Software screenshots, web page interfaces, mobile app mockups, wireframes, UI components.
- MULTIMODAL: Mixed rich documents (e.g., multi-page PDFs containing text, charts, diagrams, and photographic imagery).

Critical Invariants:
1. Return exactly ONE vision requirement from the allowed list above.
2. Do NOT perform image understanding or answer the user's visual question.
3. Do NOT generate conversational responses. ONLY output the decision.
4. Output valid JSON in the format:
   {"decision": "<EXACT_VISION_REQUIREMENT>", "confidence": <float>, "reasoning": "<string>"}
"""


class VisionRequirementDecider(BaseDecisionModule):
    """Production-ready Vision Requirement Decider powered by Strands Decider 2B.

    Runs after ContextSelectionDecider (Order 7).
    Determines whether multimodal processing is required and the specific vision modality.
    """

    ALLOWED_VISION_REQUIREMENTS: list[str] = [req.value for req in VisionRequirementType]

    def __init__(
        self,
        client: StrandsDeciderClientProtocol | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(
            module_id="decider_07_vision",
            name="Vision Requirement Decider",
            decision_type=DecisionType.CUSTOM,
            order=7,
        )
        self.settings = settings or get_settings()
        self.client = client or StrandsDeciderClient(settings=self.settings)

    def _build_contextual_prompt(self, state: CortexState) -> str:
        """Assembles rich context from preceding decisions and attachment metadata."""
        intent_info = f"Intent: {state.intent}" if state.intent else "Intent: Unspecified"
        agent_info = f"Assigned Agent: {state.selected_agent}" if state.selected_agent else "Assigned Agent: Unspecified"
        privacy_info = f"Privacy Level: {state.privacy_level.value if state.privacy_level else 'Unspecified'}"
        risk_info = f"Risk Level: {state.risk_level.value if state.risk_level else 'Unspecified'}"
        tools_info = f"Selected Tools: {state.selected_tools}" if state.selected_tools else "Selected Tools: []"
        context_info = f"Selected Context: {state.selected_context}" if state.selected_context else "Selected Context: []"

        # Check for attachment metadata
        attachment_details: list[str] = []
        if state.metadata:
            for key in ("attachments", "images", "files", "attachment_metadata", "media"):
                if key in state.metadata:
                    attachment_details.append(f"{key}={state.metadata[key]}")

        attachment_str = (
            f"Attachment Metadata: [{', '.join(attachment_details)}]\n"
            if attachment_details
            else "Attachment Metadata: [None]\n"
        )

        return (
            f"Preceding Pipeline Decisions:\n"
            f"- {intent_info}\n"
            f"- {agent_info}\n"
            f"- {privacy_info}\n"
            f"- {risk_info}\n"
            f"- {tools_info}\n"
            f"- {context_info}\n"
            f"{attachment_str}"
            f"User Prompt to Evaluate for Vision Requirement:\n"
            f"{state.user_input}"
        )

    async def _compute_decision(self, state: CortexState) -> DecisionOutcome:
        """Invokes Strands Decider 2B to evaluate required visual capability.

        Args:
            state: Current CortexState with upstream decisions.

        Returns:
            DecisionOutcome specifying the exact vision requirement.

        Raises:
            DeciderInferenceError: On malformed output, unapproved classification, or missing fields.
            DeciderTimeoutError: On inference timeout.
        """
        logger.info(
            "Evaluating vision requirement with Strands Decider 2B for request_id=%s [intent='%s', tools=%s]",
            state.request_id,
            state.intent,
            state.selected_tools,
        )

        contextual_input = self._build_contextual_prompt(state)
        start_time = time.perf_counter()

        try:
            raw_response = await self.client.decide(
                task="vision_requirement",
                user_input=contextual_input,
                allowed_choices=self.ALLOWED_VISION_REQUIREMENTS,
                system_instruction=VISION_REQUIREMENT_SYSTEM_INSTRUCTION,
            )
        except (DeciderTimeoutError, DeciderInferenceError):
            raise
        except Exception as exc:
            logger.error("Unexpected error in Strands Decider 2B VisionRequirementDecider: %s", exc)
            raise DeciderInferenceError(
                f"Strands Decider 2B vision requirement evaluation failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        decision_raw = raw_response.get("decision")
        if decision_raw is None:
            # Fallback check for "vision_requirement" key
            decision_raw = raw_response.get("vision_requirement")

        confidence_raw = raw_response.get("confidence")
        reasoning_raw = raw_response.get("reasoning", "No reasoning provided by Strands Decider 2B.")

        if decision_raw is None:
            raise DeciderInferenceError(
                "Strands Decider 2B output missing required 'decision' key in vision requirement evaluation."
            )

        decision_str = str(decision_raw).strip().upper()

        if decision_str not in self.ALLOWED_VISION_REQUIREMENTS:
            raise DeciderInferenceError(
                f"Strands Decider 2B returned unknown vision requirement '{decision_raw}'. "
                f"Must be exactly one of: {self.ALLOWED_VISION_REQUIREMENTS}"
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

        vision_type = VisionRequirementType(decision_str)

        result = VisionDecisionResult(
            vision_requirement=vision_type,
            confidence=confidence,
            reasoning=str(reasoning_raw),
            latency_ms=round(latency_ms, 2),
        )

        logger.info(
            "Strands Decider 2B determined vision requirement=%s (confidence=%.2f) in %.2fms",
            result.vision_requirement.value,
            result.confidence,
            result.latency_ms,
        )

        return DecisionOutcome(
            module_id=self.module_id,
            decision_type=self.decision_type,
            action=result.vision_requirement.value,
            confidence=result.confidence,
            reasoning=result.reasoning,
            metadata={
                "vision_requirement": result.vision_requirement.value,
                "vision_required": result.vision_requirement != VisionRequirementType.NONE,
                "latency_ms": result.latency_ms,
                "model_name": self.settings.strands.model_name,
            },
        )

    def apply_to_state(self, state: CortexState, outcome: DecisionOutcome) -> None:
        """Updates state.vision_requirement and state.vision_required."""
        vision_val = outcome.metadata.get("vision_requirement") or outcome.action
        state.vision_requirement = VisionRequirementType(vision_val)
        state.vision_required = state.vision_requirement != VisionRequirementType.NONE


# Alias to maintain backwards compatibility with existing imports
VisionDecider = VisionRequirementDecider
