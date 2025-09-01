# Tool package - MCP tool framework with state management
"""
Framework for creating custom MCP tools with automatic state management
and CPU-bound operation support.
"""

from .base import BaseTool
from .decorator import tool

__all__ = ["BaseTool", "tool"]