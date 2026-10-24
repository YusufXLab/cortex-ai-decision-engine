"""Application Layer: Orchestration, Use Cases, and Multi-Agent Workflows.

This layer coordinates the domain protocols:
1. Gathers context and calls Strands Decider 2B for decisions.
2. Applies business rules and policy decisions.
3. If approved, delegates response generation to Amazon Bedrock.
"""
