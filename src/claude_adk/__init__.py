# Claude Agent Development Kit
"""
Claude Agent Development Kit (claude-adk) - Framework for building and testing
Claude Code agents with custom MCP tools.
"""

from .agent import Agent
from .tool import BaseTool, tool

__version__ = "0.1.1"
__all__ = ["Agent", "BaseTool", "tool"]