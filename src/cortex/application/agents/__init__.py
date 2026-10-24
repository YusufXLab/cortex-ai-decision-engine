"""AI Agents package.

Contains specialized internal agents and the extensible AgentRegistry.
"""

from cortex.application.agents.registry import AgentRegistry, get_default_agent_registry

__all__ = ["AgentRegistry", "get_default_agent_registry"]
