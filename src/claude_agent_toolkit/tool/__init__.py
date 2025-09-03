# Tool package - MCP tool framework
"""
Framework for creating custom MCP tools with parallel execution support.
Users manage their own data explicitly - no automatic state management.
"""

from .base import BaseTool
from .decorator import tool

__all__ = ["BaseTool", "tool"]