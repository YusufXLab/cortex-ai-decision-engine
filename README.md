# Cortex AI Decision Engine

[![CI](https://github.com/your-org/cortex-ai-decision-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/cortex-ai-decision-engine/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9.svg)](https://docs.astral.sh/uv/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A high-performance, asynchronous AI Decision Engine built on Clean Architecture principles. It uses **Strands Decider 2B** to make intelligent routing, policy, and classification decisions *before* delegating response generation to **Amazon Bedrock**.

---

## 🏛️ Fundamental Architectural Invariant

| Plane | Component | Strict Responsibility | Prohibited Actions |
|---|---|---|---|
| **Decision Plane** | **Strands Decider 2B** | Evaluates inputs, classifies intent, verifies guardrails, and produces routing decisions. | **NEVER** generates user-facing responses. |
| **Generation Plane** | **Amazon Bedrock** | Generates rich textual or multimodal completions based on decider context. | **NEVER** performs routing or policy evaluation. |

This invariant decouples lightweight, deterministic, low-latency decision trees from expensive foundation model generation, cutting API costs, reducing latency, and enhancing security.

---

## 📁 Project Directory Structure & Rationale

```text
cortex-ai-decision-engine/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: Ruff linting, Black formatting, Pytest on Python 3.12
├── src/
│   └── cortex/
│       ├── core/                 # Cross-cutting concerns & foundational utilities
│       │   ├── config.py         # Type-safe settings via Pydantic Settings v2
│       │   ├── exceptions.py     # Domain exception hierarchy (Decider/Bedrock/Orchestrator)
│       │   └── logging.py        # Structured JSON logging & correlation ID tracing
│       │
│       ├── domain/               # Pure Enterprise Business Rules (Zero external dependencies)
│       │   ├── models/           # Domain entities (DecisionOutcome, GenerationInput, etc.)
│       │   │   ├── decision.py   # Decision plane data structures
│       │   │   └── generation.py # Generation plane data structures
│       │   └── protocols/        # PEP 544 abstract protocols (Decider, Bedrock, Orchestrator)
│       │       ├── bedrock.py
│       │       ├── decider.py
│       │       └── orchestrator.py
│       │
│       ├── application/          # Use Cases & Application Orchestration
│       │   ├── orchestrators/    # Central Decision Orchestrator (coordinates the 11 modules)
│       │   ├── agents/           # Specialized multi-agent workflows
│       │   └── services/         # Application pipeline services & telemetry helpers
│       │
│       ├── infrastructure/       # Concrete external adapters & drivers
│       │   ├── deciders/         # Implementations of the 11 Strands Decider 2B modules
│       │   ├── bedrock/          # AWS Bedrock asynchronous boto3/aioboto3 adapters
│       │   └── benchmarking/     # Latency, accuracy, and cost tracking harness
│       │
│       ├── presentation/         # Driving adapters (REST API / Transport layer)
│       │   ├── api/
│       │   │   ├── deps.py       # FastAPI dependency injection providers
│       │   │   └── v1/
│       │   │       ├── router.py # Master API v1 router
│       │   │       └── endpoints/
│       │   │           ├── health.py # Health, liveness, and readiness probes
│       │   │           └── demo.py   # Demo endpoint scaffold for hackathon testing
│       │   ├── middleware/       # Correlation ID & request latency timing middleware
│       │   └── schemas/          # API Request/Response DTOs and unified API envelope
│       │
│       └── main.py               # FastAPI application factory, lifespan, and exception handlers
│
├── tests/
│   ├── conftest.py               # Shared pytest fixtures & async HTTPX client
│   ├── unit/                     # Fast isolated unit tests (config, domain entities)
│   └── integration/              # End-to-end API & middleware integration tests
│
├── .dockerignore
├── .env.example                  # Environment variable reference template
├── .gitignore
├── docker-compose.yml            # Local development orchestration with live reloading
├── Dockerfile                    # Multi-stage production container image with uv
├── pyproject.toml                # Project metadata, dependencies, Ruff/Black/Pytest config
└── README.md
```

### Why Each Folder Exists (Clean Architecture & SOLID)

1. **`src/cortex/domain/`**:
   - **Purpose**: Represents the core business concepts of our decision engine.
   - **SOLID**: Follows the **Single Responsibility Principle (SRP)** and **Dependency Inversion Principle (DIP)**. It contains zero references to FastAPI, AWS SDKs, or databases. Any external framework change will never affect the domain.
   - **Protocols (`protocols/`)**: Defines abstract PEP 544 structural subtyping contracts (`DeciderProtocol`, `BedrockClientProtocol`). High-level modules depend on these abstractions, not on concrete AWS SDKs or model weights.

2. **`src/cortex/core/`**:
   - **Purpose**: Provides cross-cutting platform concerns including type-safe configuration (`Pydantic Settings v2`), correlation-aware structured logging, and an expressive domain exception tree.

3. **`src/cortex/application/`**:
   - **Purpose**: Implements the use cases.
   - **`orchestrators/`**: Reserved for the upcoming **Decision Orchestrator**, which executes the 11 decision modules in parallel or sequence, evaluates confidence thresholds, and decides if Bedrock should be invoked.
   - **`agents/`**: Reserved for multi-agent workflows that consult the Decider before taking actions.
   - **`services/`**: Coordinates persistence, caching, and pipeline dispatch.

4. **`src/cortex/infrastructure/`**:
   - **Purpose**: Implements the technical details.
   - **`deciders/`**: Where the **11 Strands Decider 2B** modules will live.
   - **`bedrock/`**: Where the AWS Bedrock client adapter resides (managing boto3 sessions, retries, exponential backoff, and credentials).
   - **`benchmarking/`**: Measures latency profiles, comparing raw LLM calls vs. Decider-gated calls.

5. **`src/cortex/presentation/`**:
   - **Purpose**: The driving adapter layer (FastAPI HTTP endpoints, DTO schemas, and HTTP middleware). It adapts incoming HTTP requests to domain models and formats domain results into standard JSON envelopes.

---

## ⚡ Tech Stack

- **Python 3.12**: Leveraging modern features (e.g. `type` statements, enhanced asyncio performance, strict typing).
- **Astral `uv`**: Ultra-fast Rust-based Python package manager and resolver.
- **FastAPI**: Asynchronous web framework with automatic OpenAPI docs.
- **Pydantic v2**: High-speed Rust-based data validation and settings management.
- **AWS Bedrock / Boto3 / aioBoto3**: Fully async AWS foundation model invocations.
- **Ruff & Black**: Modern linting, formatting, and code standards enforcement.
- **Pytest & Pytest-Asyncio**: Asynchronous test runner with coverage reporting.
- **Docker**: Multi-stage, non-root, cache-optimized containerization.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites

Ensure you have **Python 3.12+** and **uv** installed:

```bash
# On Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Environment Setup

Clone the repository and copy the environment template:

```bash
cp .env.example .env
```

### 3. Install Dependencies

Using `uv`, sync the environment and install dev extras:

```bash
uv sync --all-extras --dev
```

### 4. Run the Development Server

```bash
uv run uvicorn cortex.main:app --host 0.0.0.0 --port 8000 --reload
```

The server is available at:
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- **Demo Pipeline Scaffold**: `POST http://localhost:8000/api/v1/demo/decide-and-generate`

---

## 🐳 Docker & Docker Compose

### Run via Docker Compose

```bash
docker compose up --build
```

### Standalone Docker Build

```bash
docker build -t cortex-ai-decision-engine:latest .
docker run -p 8000:8000 --env-file .env cortex-ai-decision-engine:latest
```

---

## 🧪 Testing, Quality & CI/CD

### Run Pytest Suite

```bash
# Run tests with coverage
uv run pytest --cov=src/cortex --cov-report=term-missing
```

### Run Linter & Formatter

```bash
# Check code style with Ruff
uv run ruff check .

# Check code formatting with Black
uv run black --check .

# Automatically apply Ruff fixes & Black formatting
uv run ruff check --fix .
uv run black .
```

---

## 🧠 First AI Decision Module: IntentDecider (Strands Decider 2B)

`IntentDecider` is the first production-ready AI decision module in the Cortex AI Decision Engine. It uses **Strands Decider 2B** to determine the single best intent category without any hardcoded rule matching or frontier model generation.

### Why Strands Decider 2B is Used
- **Sub-50ms Latency**: Frontier models (Claude 3.5 Sonnet, GPT-4) take 800ms-2s just to classify intent. Strands Decider 2B evaluates semantic intent in tens of milliseconds.
- **Cost Efficiency**: Frontier model token pricing is avoided for routing decisions.
- **Strict Invariant**: Strands Decider 2B **ONLY** makes decisions; it never generates output responses.

### Strictly Allowed Intent Categories
Strands Decider 2B evaluates the query and returns **exactly ONE** category:
1. `CHAT`: Casual conversation, chit-chat, greetings.
2. `WRITING`: Composition, prose, essays, drafts, or editing.
3. `CODING`: Programming, software engineering, debugging, scripts, or cloud architecture.
4. `RESEARCH`: Deep factual discovery, external information synthesis, literature inquiries.
5. `ANALYSIS`: Comparative critiques, numerical/risk evaluations, financial or logical assessments.
6. `PLANNING`: Sequential roadmaps, action plans, itineraries, sprint backlogs.
7. `VISION`: Multimodal image comprehension, OCR, diagram reading.

### Developer Reusability Pattern
To implement any of the remaining 10 decision modules (e.g. `PrivacyDecider`, `RiskDecider`, `ComplexityDecider`):
1. Inherit from [`BaseDecisionModule`](src/cortex/infrastructure/deciders/base.py).
2. Set unique `module_id`, `name`, `decision_type`, and execution `order` (1–11).
3. Inject [`StrandsDeciderClientProtocol`](src/cortex/domain/protocols/strands_client.py) via `__init__`.
4. Implement `_compute_decision(state: CortexState)`:
   - Call `await self.client.decide(...)` with task name and permitted categorical options.
   - Validate model output strictly (confidence, allowed values).
   - On error or invalid output, raise domain exception (`DeciderInferenceError`, `DeciderTimeoutError`).
5. Implement `apply_to_state(state: CortexState, outcome: DecisionOutcome)` to mutate the corresponding field on `CortexState`.
6. Add the module to [`get_default_decision_modules()`](src/cortex/infrastructure/deciders/registry.py).

---

## 🤖 Second AI Decision Module: AgentSelectionDecider (Strands Decider 2B)

`AgentSelectionDecider` is the second production AI decision module in the pipeline. It answers a fundamentally different architectural question from Intent Classification:

| Decision Step | Core Question Answered | Example Decisions |
|---|---|---|
| **Step 1: IntentDecider** | *"What is the user trying to do at a high level?"* | `CODING`, `RESEARCH`, `WRITING`, `ANALYSIS` |
| **Step 2: AgentSelectionDecider** | *"Which specialized internal agent is best suited to solve this specific request?"* | `DevOpsAgent`, `DebuggingAgent`, `RefactoringAgent` |

### Why Agent Selection is a Separate AI Decision from Intent Classification
A simplistic 1-to-1 mapping between Intent and Agent renders a multi-agent system redundant. Under the **SAME** intent (e.g. `CODING`), different technical tasks require completely different tools, domain context, and agent personas:
- **`Intent = CODING`**:
  - *"Why is my async task hanging with deadlock?"* $\rightarrow$ **`DebuggingAgent`** (root cause analysis, stack traces)
  - *"Deploy this FastAPI container to AWS ECS Fargate with Terraform."* $\rightarrow$ **`DevOpsAgent`** (cloud architecture, Docker, CI/CD)
  - *"Refactor this 600-line class to follow Single Responsibility."* $\rightarrow$ **`RefactoringAgent`** (code restructuring, modularity)
  - *"Audit this AES-GCM encryption code for security vulnerabilities."* $\rightarrow$ **`CodeReviewAgent`** (security auditing, quality gates)
  - *"Implement an AVL balanced binary tree in Python."* $\rightarrow$ **`SoftwareEngineerAgent`** (algorithm implementation)
- **`Intent = RESEARCH`**:
  - Academic papers and citations $\rightarrow$ **`LiteratureReviewAgent`**
  - Real-time internet facts & market data $\rightarrow$ **`WebResearchAgent`**
  - Claim validation & source checking $\rightarrow$ **`FactCheckingAgent`**
  - Hardware & RFC protocol whitepapers $\rightarrow$ **`TechnicalResearchAgent`**
- **`Intent = WRITING`**:
  - Executive & client correspondence $\rightarrow$ **`EmailWriterAgent`**
  - API documentation & developer guides $\rightarrow$ **`TechnicalWriterAgent`**
  - Landing page copy & announcements $\rightarrow$ **`MarketingCopyAgent`**
- **`Intent = ANALYSIS`**:
  - Threat modeling & migration risk $\rightarrow$ **`RiskAnalysisAgent`**
  - Unit economics, cloud ROI & valuation $\rightarrow$ **`FinancialAnalysisAgent`**
  - Statistical datasets & numerical trends $\rightarrow$ **`DataAnalysisAgent`**
- **`Intent = VISION`**:
  - Scanned text, receipts, and table extraction $\rightarrow$ **`OCRAgent`**
  - System architecture schemas & UML flowcharts $\rightarrow$ **`DiagramAnalysisAgent`**

### Dynamic Extensibility via `AgentRegistry`
To add a new specialized agent without modifying the orchestrator or `AgentSelectionDecider`:
```python
from cortex.application.agents.registry import get_default_agent_registry
from cortex.domain.models.agent import AgentProfile
from cortex.domain.models.intent import IntentCategory

registry = get_default_agent_registry()
registry.register(
    AgentProfile(
        name="QuantumCryptographyAgent",
        intents=[IntentCategory.CODING, IntentCategory.RESEARCH],
        description="Specialized in post-quantum lattice cryptography and Shor's algorithm.",
        capabilities=["post_quantum_crypto", "qiskit", "lattice_algorithms"],
    )
)
```
Strands Decider 2B dynamically considers the newly registered agent profile when evaluating requests matching those intents!

---

## 🗺️ Roadmap: Ready for Upcoming Additions

This foundation is architected to seamlessly accommodate the next implementation phases without structural refactoring:

```mermaid
graph TD
    Client([HTTP Client / SDK]) -->|1. Request| Presentation[Presentation Layer / FastAPI]
    Presentation -->|2. DTO -> Domain| AppOrchestrator[Decision Orchestrator]
    
    subgraph "Decision Plane (Strands Decider 2B)"
        AppOrchestrator -->|3. Evaluate in parallel| Deciders[11 Decider Modules]
        Deciders -.->|Intent, Guardrail, Complexity, Routing| Deciders
        Deciders -->|4. Decision Outcomes| AppOrchestrator
    end

    subgraph "Gating & Policy Evaluation"
        AppOrchestrator -->|5. Gate Check| Gate{Approved & Confidence >= 0.75?}
    end

    Gate -->|Rejected / Fallback| EarlyExit[Instant Direct Response / Safety Block]
    
    subgraph "Generation Plane (Amazon Bedrock)"
        Gate -->|6. Approved| BedrockClient[Bedrock Adapter]
        BedrockClient -->|7. Generate Response| Claude[Claude 3.5 Sonnet / Bedrock FM]
        Claude -->|8. Generated Tokens| BedrockClient
        BedrockClient -->|9. Generation Outcome| AppOrchestrator
    end

    AppOrchestrator -->|10. Full Telemetry & Response| Presentation
    Presentation -->|11. APIResponse| Client
```

1. **11 Strands Decider 2B Modules**:
   Implement concrete classes in `src/cortex/infrastructure/deciders/` satisfying `DeciderProtocol` (e.g., `intent_decider.py`, `guardrail_decider.py`, `complexity_decider.py`, `routing_decider.py`, etc.).
2. **Decision Orchestrator**:
   Implement in `src/cortex/application/orchestrators/` to fan-out decision requests across modules using `asyncio.gather()`, apply weightings, and gate calls to Bedrock.
3. **Multiple AI Agents**:
   Add specialized agents in `src/cortex/application/agents/` inheriting from domain agent protocols.
4. **Amazon Bedrock Integration**:
   Implement asynchronous client in `src/cortex/infrastructure/bedrock/` wrapping Bedrock Converse API / InvokeModel.
5. **Benchmarking Harness**:
   Populate `src/cortex/infrastructure/benchmarking/` with latency analyzers and cost comparison calculators.
