"""Extensible Agent Registry for specialized internal agents.

Enables the Open/Closed Principle (OCP):
New specialized agents can be registered at runtime or module load time
without modifying the DecisionOrchestrator or AgentSelectionDecider.
"""

from cortex.domain.models.agent import AgentProfile
from cortex.domain.models.intent import IntentCategory


class AgentRegistry:
    """Registry maintaining specialized agent profiles and their intent mappings."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentProfile] = {}

    def register(self, profile: AgentProfile) -> None:
        """Register a new specialized agent profile."""
        self._agents[profile.name] = profile

    def get_agent(self, name: str) -> AgentProfile | None:
        """Look up an agent profile by name."""
        return self._agents.get(name)

    def get_candidates(self, intent: IntentCategory | str | None) -> list[AgentProfile]:
        """Returns candidate agent profiles eligible for the given intent.

        If intent is unknown or has no specific matches, returns all agents or general fallbacks.
        """
        if not intent:
            return list(self._agents.values())

        # Normalize string or enum
        intent_value = intent.value if isinstance(intent, IntentCategory) else str(intent).upper()

        candidates = [
            agent
            for agent in self._agents.values()
            if any(i.value == intent_value for i in agent.intents)
        ]

        # If no specialized match found, fallback to all registered agents
        return candidates if candidates else list(self._agents.values())

    def list_agents(self) -> list[AgentProfile]:
        """Returns all registered agent profiles."""
        return list(self._agents.values())


def get_default_agent_registry() -> AgentRegistry:
    """Constructs the default registry populated with domain specialized agents."""
    registry = AgentRegistry()

    # ==========================================================================
    # 1. CODING Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="SoftwareEngineerAgent",
            intents=[IntentCategory.CODING],
            description="Specialized in implementing new features, writing clean production code, and algorithms.",
            capabilities=["feature_development", "python", "typescript", "clean_architecture"],
        )
    )
    registry.register(
        AgentProfile(
            name="DebuggingAgent",
            intents=[IntentCategory.CODING],
            description="Specialized in diagnosing errors, root cause analysis, stack traces, and fixing bugs.",
            capabilities=["bug_fixing", "root_cause_analysis", "exception_handling"],
        )
    )
    registry.register(
        AgentProfile(
            name="CodeReviewAgent",
            intents=[IntentCategory.CODING],
            description="Specialized in auditing code quality, security vulnerabilities, and adherence to style standards.",
            capabilities=["static_analysis", "security_audit", "style_compliance"],
        )
    )
    registry.register(
        AgentProfile(
            name="RefactoringAgent",
            intents=[IntentCategory.CODING],
            description="Specialized in restructuring existing code, reducing complexity, and improving maintainability.",
            capabilities=["design_patterns", "modularity", "technical_debt_reduction"],
        )
    )
    registry.register(
        AgentProfile(
            name="DevOpsAgent",
            intents=[IntentCategory.CODING],
            description="Specialized in cloud deployment, Docker, ECS, Kubernetes, CI/CD pipelines, and infrastructure as code.",
            capabilities=["docker", "aws_ecs", "terraform", "github_actions", "cloud_architecture"],
        )
    )

    # ==========================================================================
    # 2. RESEARCH Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="LiteratureReviewAgent",
            intents=[IntentCategory.RESEARCH],
            description="Specialized in academic literature, scholarly publications, paper synthesis, and citations.",
            capabilities=["academic_search", "citation_analysis", "paper_summarization"],
        )
    )
    registry.register(
        AgentProfile(
            name="WebResearchAgent",
            intents=[IntentCategory.RESEARCH],
            description="Specialized in real-time web search, market intelligence, industry trends, and public data.",
            capabilities=["web_search", "market_intelligence", "competitive_analysis"],
        )
    )
    registry.register(
        AgentProfile(
            name="FactCheckingAgent",
            intents=[IntentCategory.RESEARCH],
            description="Specialized in verifying claims, validating sources, and cross-referencing truthfulness.",
            capabilities=["source_verification", "claim_checking", "hallucination_prevention"],
        )
    )
    registry.register(
        AgentProfile(
            name="TechnicalResearchAgent",
            intents=[IntentCategory.RESEARCH],
            description="Specialized in deep RFC specs, hardware architectures, standard protocols, and technical whitepapers.",
            capabilities=["rfc_specifications", "protocol_analysis", "hardware_specs"],
        )
    )

    # ==========================================================================
    # 3. WRITING Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="EmailWriterAgent",
            intents=[IntentCategory.WRITING],
            description="Specialized in professional correspondence, customer outreach, negotiation emails, and executive messaging.",
            capabilities=["business_email", "tone_adaptation", "client_communication"],
        )
    )
    registry.register(
        AgentProfile(
            name="TechnicalWriterAgent",
            intents=[IntentCategory.WRITING],
            description="Specialized in API reference documentation, engineering guides, and technical tutorials.",
            capabilities=["api_docs", "tutorials", "developer_guides"],
        )
    )
    registry.register(
        AgentProfile(
            name="MarketingCopyAgent",
            intents=[IntentCategory.WRITING],
            description="Specialized in persuasive marketing copy, product launches, landing pages, and value propositions.",
            capabilities=["copywriting", "social_media", "conversion_optimization"],
        )
    )
    registry.register(
        AgentProfile(
            name="DocumentationAgent",
            intents=[IntentCategory.WRITING],
            description="Specialized in system architecture documentation, READMEs, changelogs, and codebase wikis.",
            capabilities=["readme_generation", "changelogs", "architecture_docs"],
        )
    )

    # ==========================================================================
    # 4. ANALYSIS Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="DataAnalysisAgent",
            intents=[IntentCategory.ANALYSIS],
            description="Specialized in statistical metrics, dataset exploration, distributions, and data-driven insights.",
            capabilities=["statistical_analysis", "pandas", "trend_detection"],
        )
    )
    registry.register(
        AgentProfile(
            name="RiskAnalysisAgent",
            intents=[IntentCategory.ANALYSIS],
            description="Specialized in risk assessment, threat modeling, security evaluations, and migration trade-offs.",
            capabilities=["threat_modeling", "cloud_migration_risk", "security_assessment"],
        )
    )
    registry.register(
        AgentProfile(
            name="FinancialAnalysisAgent",
            intents=[IntentCategory.ANALYSIS],
            description="Specialized in financial statement modeling, unit economics, budget forecasting, and cost analysis.",
            capabilities=["valuation", "roi_calculation", "cloud_cost_economics"],
        )
    )

    # ==========================================================================
    # 5. PLANNING Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="SprintPlanningAgent",
            intents=[IntentCategory.PLANNING],
            description="Specialized in agile sprint backlog planning, story point estimation, and task decomposition.",
            capabilities=["agile_sprints", "jira_breakdown", "estimation"],
        )
    )
    registry.register(
        AgentProfile(
            name="ArchitecturePlanningAgent",
            intents=[IntentCategory.PLANNING],
            description="Specialized in multi-phase system architecture roadmaps, technical milestones, and dependencies.",
            capabilities=["system_milestones", "migration_phases", "dependency_mapping"],
        )
    )
    registry.register(
        AgentProfile(
            name="ProjectManagementAgent",
            intents=[IntentCategory.PLANNING],
            description="Specialized in resource allocation, Gantt schedules, risk mitigation plans, and stakeholder updates.",
            capabilities=["timeline_management", "resource_allocation", "critical_path"],
        )
    )

    # ==========================================================================
    # 6. VISION Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="OCRAgent",
            intents=[IntentCategory.VISION],
            description="Specialized in extracting tabular data, text, receipts, and structured fields from images and documents.",
            capabilities=["ocr", "table_extraction", "receipt_parsing"],
        )
    )
    registry.register(
        AgentProfile(
            name="DiagramAnalysisAgent",
            intents=[IntentCategory.VISION],
            description="Specialized in parsing system architecture diagrams, UML charts, database schemas, and flowcharts.",
            capabilities=["uml_parsing", "architecture_diagrams", "schema_visualization"],
        )
    )
    registry.register(
        AgentProfile(
            name="ImageUnderstandingAgent",
            intents=[IntentCategory.VISION],
            description="Specialized in open-ended visual perception, scene understanding, object detection, and visual QA.",
            capabilities=["scene_description", "object_recognition", "visual_qa"],
        )
    )

    # ==========================================================================
    # 7. CHAT Intent Agents
    # ==========================================================================
    registry.register(
        AgentProfile(
            name="GeneralAssistantAgent",
            intents=[IntentCategory.CHAT],
            description="General-purpose conversational assistant for open-ended queries, assistance, and guidance.",
            capabilities=["general_qa", "conversational_dialogue"],
        )
    )
    registry.register(
        AgentProfile(
            name="TutorAgent",
            intents=[IntentCategory.CHAT],
            description="Specialized in educational tutoring, Socratic dialogue, step-by-step concept explanations, and study guides.",
            capabilities=["socratic_tutoring", "concept_explanations", "pedagogical_guides"],
        )
    )

    return registry
