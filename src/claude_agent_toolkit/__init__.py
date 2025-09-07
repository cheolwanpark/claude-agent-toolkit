# Claude Agent Toolkit
"""
Claude Agent Toolkit (claude-agent-toolkit) - Framework for building and testing
Claude Code agents with custom MCP tools.
"""

from .agent import Agent
from .tool import BaseTool, tool, ToolInfo, list_tools
from .logging import set_logging, LogLevel
from .exceptions import (
    ClaudeAgentError,
    ConfigurationError, 
    ConnectionError,
    ExecutionError
)

__version__ = "0.1.3b5"
__all__ = [
    "Agent", 
    "BaseTool", 
    "tool", 
    "ToolInfo",
    "list_tools",
    "set_logging", 
    "LogLevel",
    "ClaudeAgentError",
    "ConfigurationError", 
    "ConnectionError",
    "ExecutionError"
]