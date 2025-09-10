# Claude Agent Toolkit
"""
Claude Agent Toolkit (claude-agent-toolkit) - Framework for building and testing
Claude Code agents with custom MCP tools.
"""

from .agent import Agent
from .agent.executor import ExecutorType
from .tool import BaseTool, tool, ToolInfo, list_tools
from .logging import set_logging, LogLevel
from .exceptions import (
    ClaudeAgentError,
    ConfigurationError, 
    ConnectionError,
    ExecutionError
)

__version__ = "0.2.1"
__all__ = [
    "Agent", 
    "ExecutorType",
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