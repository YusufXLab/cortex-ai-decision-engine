"""Asynchronous HTTP client for invoking Strands Decider 2B.

Strict Invariant:
Strands Decider 2B is used strictly for low-latency decision-making,
classification, and routing. It NEVER generates responses for the user.
"""

from typing import Any
import httpx
from cortex.core.config import Settings, get_settings
from cortex.core.exceptions import (
    DeciderInferenceError,
    DeciderTimeoutError,
)
from cortex.core.logging import get_logger
from cortex.domain.protocols.strands_client import StrandsDeciderClientProtocol

logger = get_logger("cortex.deciders.client")


class StrandsDeciderClient(StrandsDeciderClientProtocol):
    """Production client for communicating with Strands Decider 2B model endpoints.

    Supports both live HTTP inference (vLLM / Triton / Ollama / FastAPI daemon)
    and offline mock simulation for development and testing.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.endpoint = self.settings.strands.endpoint
        self.timeout_seconds = self.settings.strands.timeout_seconds
        self.model_name = self.settings.strands.model_name
        self.mock_mode = self.settings.strands.mock_mode
        self._http_client = http_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is not None:
            return self._http_client
        return httpx.AsyncClient(timeout=self.timeout_seconds)

    async def decide(
        self,
        task: str,
        user_input: str,
        allowed_choices: list[str],
        system_instruction: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sends a decision prompt to Strands Decider 2B.

        Args:
            task: Task classification identifier.
            user_input: User text or context to evaluate.
            allowed_choices: Strictly allowed output categories.
            system_instruction: System prompt framing the decision criteria.
            parameters: Optional inference parameters (e.g. temperature).

        Returns:
            Dict containing 'decision', 'confidence', and 'reasoning'.

        Raises:
            DeciderTimeoutError: On inference timeout.
            DeciderInferenceError: On network failure, non-200 HTTP, or malformed response.
        """
        if self.mock_mode:
            return await self._mock_decision(task, user_input, allowed_choices)

        payload = {
            "model": self.model_name,
            "task": task,
            "input": user_input,
            "allowed_choices": allowed_choices,
            "system_instruction": system_instruction,
            "parameters": parameters or {"temperature": 0.0, "max_tokens": 150},
        }

        logger.debug(
            "Invoking Strands Decider 2B at %s for task='%s'",
            self.endpoint,
            task,
        )

        should_close = self._http_client is None
        client = await self._get_client()

        try:
            response = await client.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict) or "decision" not in data:
                raise DeciderInferenceError(
                    f"Strands Decider 2B returned malformed output structure: {data}"
                )

            return data

        except httpx.TimeoutException as exc:
            logger.error("Strands Decider 2B timed out after %.1fs: %s", self.timeout_seconds, exc)
            raise DeciderTimeoutError(
                f"Strands Decider 2B request timed out after {self.timeout_seconds}s"
            ) from exc

        except httpx.HTTPStatusError as exc:
            logger.error("Strands Decider 2B returned HTTP %d: %s", exc.response.status_code, exc.response.text)
            raise DeciderInferenceError(
                f"Strands Decider 2B returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

        except httpx.RequestError as exc:
            logger.error("Failed to connect to Strands Decider 2B: %s", exc)
            raise DeciderInferenceError(
                f"Connection to Strands Decider 2B failed at {self.endpoint}: {exc}"
            ) from exc

        finally:
            if should_close:
                await client.aclose()

    async def _mock_decision(
        self,
        task: str,
        user_input: str,
        allowed_choices: list[str],
    ) -> dict[str, Any]:
        """Realistic simulated model decision for offline development."""
        lowered = user_input.lower()

        if task == "agent_selection":
            # Semantic agent selection simulation
            selected = allowed_choices[0]
            reasoning = "Best matching specialized agent for request parameters."

            if any(k in lowered for k in ["deploy", "ecs", "docker", "cluster", "kubernetes", "pipeline"]) and "DevOpsAgent" in allowed_choices:
                selected = "DevOpsAgent"
                reasoning = "Request demands cloud deployment, container orchestration, and infrastructure engineering."
            elif any(k in lowered for k in ["bug", "fix", "crash", "error", "trace"]) and "DebuggingAgent" in allowed_choices:
                selected = "DebuggingAgent"
                reasoning = "Request requires diagnostic root cause analysis and software defect resolution."
            elif any(k in lowered for k in ["refactor", "restructure", "clean"]) and "RefactoringAgent" in allowed_choices:
                selected = "RefactoringAgent"
                reasoning = "Request asks for code reorganization and technical debt reduction without altering behavior."
            elif any(k in lowered for k in ["review", "audit", "security"]) and "CodeReviewAgent" in allowed_choices:
                selected = "CodeReviewAgent"
                reasoning = "Auditing code quality, security vulnerabilities, and adherence to guidelines."
            elif any(k in lowered for k in ["risk", "migration", "threat"]) and "RiskAnalysisAgent" in allowed_choices:
                selected = "RiskAnalysisAgent"
                reasoning = "Request involves migration risk modeling, failure modes, and security assessment."
            elif any(k in lowered for k in ["finance", "cost", "budget", "dollar", "unit economic"]) and "FinancialAnalysisAgent" in allowed_choices:
                selected = "FinancialAnalysisAgent"
                reasoning = "Quantitative financial modeling, cloud economics, or ROI analysis requested."
            elif any(k in lowered for k in ["diagram", "uml", "flowchart", "architecture chart"]) and "DiagramAnalysisAgent" in allowed_choices:
                selected = "DiagramAnalysisAgent"
                reasoning = "Parsing and interpreting visual system architecture schemas or UML diagrams."
            elif any(k in lowered for k in ["ocr", "receipt", "table", "document scan"]) and "OCRAgent" in allowed_choices:
                selected = "OCRAgent"
                reasoning = "Extracting tabular data and characters from scanned document imagery."
            elif any(k in lowered for k in ["api", "doc", "guide", "tutorial"]) and "TechnicalWriterAgent" in allowed_choices:
                selected = "TechnicalWriterAgent"
                reasoning = "Drafting comprehensive API technical documentation or engineering guide."
            elif any(k in lowered for k in ["paper", "academic", "literature", "citation"]) and "LiteratureReviewAgent" in allowed_choices:
                selected = "LiteratureReviewAgent"
                reasoning = "Synthesizing scholarly publications and academic citations."

            return {
                "decision": selected,
                "confidence": 0.95,
                "reasoning": f"[Strands Decider 2B Agent Selector]: {reasoning}",
            }

        if task == "privacy_classification":
            decision = "PUBLIC"
            reasoning = "General public domain inquiry without sensitive data."

            if any(k in lowered for k in ["medical", "health", "patient", "prescription", "diagnosis", "doctor", "symptom", "clinical", "hospital"]):
                decision = "MEDICAL"
                reasoning = "Query references protected healthcare, clinical symptoms, or patient medical records."
            elif any(k in lowered for k in ["financial", "bank", "credit card", "tax", "salary", "payroll", "revenue", "portfolio", "invoice", "wire transfer"]):
                decision = "FINANCIAL"
                reasoning = "Query contains financial transactions, banking credentials, payroll, or tax details."
            elif any(k in lowered for k in ["legal", "nda", "contract", "attorney", "lawsuit", "litigation", "clause", "non-disclosure", "settlement"]):
                decision = "LEGAL"
                reasoning = "Query involves legal contracts, non-disclosure agreements, litigation briefs, or attorney communications."
            elif any(k in lowered for k in ["ssn", "passport", "social security", "home address", "date of birth", "personal phone", "private diary"]):
                decision = "PERSONAL"
                reasoning = "Query contains personally identifiable information (PII) or sensitive personal life details."
            elif any(k in lowered for k in ["confidential", "proprietary", "internal secret", "api key", "trade secret", "unreleased merger", "source code leak"]):
                decision = "CONFIDENTIAL"
                reasoning = "Query involves proprietary business secrets, internal confidential systems, or restricted assets."

            if decision not in allowed_choices:
                decision = allowed_choices[0]

            return {
                "decision": decision,
                "confidence": 0.96,
                "reasoning": f"[Strands Decider 2B Privacy Classifier]: {reasoning}",
            }

        if task == "risk_assessment":
            decision = "LOW"
            reasoning = "Benign informational query with minimal operational risk."

            if any(k in lowered for k in ["ignore all rules", "jailbreak", "exploit", "ddos", "malware", "bypass security", "exfiltrate"]):
                decision = "CRITICAL"
                reasoning = "Detected high-threat malicious prompt injection, exploit payload, or severe safety violation."
            elif any(k in lowered for k in ["delete database", "drop table", "override prod", "rm -rf", "wipe disk", "kill process"]):
                decision = "HIGH"
                reasoning = "Destructive data mutation or destructive operational action requiring manual verification."
            elif any(k in lowered for k in ["deploy", "modify config", "alter table", "migration", "batch update", "schema change"]):
                decision = "MODERATE"
                reasoning = "Operational state change or deployment requiring step-level verification."

            if decision not in allowed_choices:
                decision = allowed_choices[0]

            return {
                "decision": decision,
                "confidence": 0.96,
                "reasoning": f"[Strands Decider 2B Risk Evaluator]: {reasoning}",
            }

        if task == "tool_selection":
            selected_tools: list[str] = []

            if any(k in lowered for k in ["search", "google", "web", "latest", "recent news", "current price", "today"]):
                selected_tools.append("WEB_SEARCH")
            if any(k in lowered for k in ["calculate", "math", "sum", "formula", "arithmetic", "variance", "percentage"]):
                selected_tools.append("CALCULATOR")
            if any(k in lowered for k in ["file", "pdf", "csv", "read doc", "document scan", "upload"]):
                selected_tools.append("FILE_READER")
            if any(k in lowered for k in ["image", "diagram", "screenshot", "chart", "ocr", "photo"]):
                selected_tools.append("VISION")
            if any(k in lowered for k in ["database", "sql", "query table", "postgres", "records"]):
                selected_tools.append("DATABASE")
            if any(k in lowered for k in ["execute", "run code", "sandbox", "python script", "run"]):
                selected_tools.append("CODE_EXECUTION")
            if any(k in lowered for k in ["remember", "past conversation", "my preference", "recall"]):
                selected_tools.append("MEMORY")
            if any(k in lowered for k in ["internal doc", "knowledge base", "rag", "embeddings"]):
                selected_tools.append("RAG")

            if not selected_tools:
                selected_tools = ["NONE"]

            # Filter by allowed choices if provided
            valid_tools = [t for t in selected_tools if t in allowed_choices] or ["NONE"]

            return {
                "decision": valid_tools,
                "confidence": 0.95,
                "reasoning": f"[Strands Decider 2B Tool Selector]: Selected capabilities {valid_tools} to fulfill task requirements.",
            }

        # Semantic routing simulation for offline testing
        if "code" in lowered or "python" in lowered or "function" in lowered or "bug" in lowered or "refactor" in lowered or "deploy" in lowered:
            decision = "CODING"
            reasoning = "Query requires software engineering, deployment, or code implementation."
        elif "research" in lowered or "investigate" in lowered or "study" in lowered or "literature" in lowered:
            decision = "RESEARCH"
            reasoning = "In-depth factual inquiry requiring literature or web research."
        elif "analyze" in lowered or "risk" in lowered or "compare" in lowered or "eval" in lowered or "metrics" in lowered:
            decision = "ANALYSIS"
            reasoning = "Analytical evaluation or risk assessment requested."
        elif "write" in lowered or "draft" in lowered or "essay" in lowered or "story" in lowered or "email" in lowered:
            decision = "WRITING"
            reasoning = "Creative or professional document composition requested."
        elif "plan" in lowered or "roadmap" in lowered or "schedule" in lowered or "steps" in lowered:
            decision = "PLANNING"
            reasoning = "Multi-step strategic planning and sequencing requested."
        elif "image" in lowered or "picture" in lowered or "diagram" in lowered or "screenshot" in lowered:
            decision = "VISION"
            reasoning = "Visual inspection or multimodal comprehension required."
        else:
            decision = "CHAT"
            reasoning = "Conversational or general inquiry."

        if decision not in allowed_choices:
            decision = allowed_choices[0]

        return {
            "decision": decision,
            "confidence": 0.96,
            "reasoning": f"[Strands Decider 2B Offline Simulation]: {reasoning}",
        }
