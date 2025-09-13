# Tool package - MCP tool framework
"""
Framework for creating custom MCP tools with parallel execution support.
Users manage their own data explicitly - no automatic state management.
"""

from .abstract import AbstractTool
from .base import BaseTool
from .decorator import tool
from .utils import ToolInfo, list_tools

# BaseTool is now the concrete HTTP-based MCP tool implementation
# AbstractTool is available for those who need the interface
__all__ = ["BaseTool", "AbstractTool", "tool", "ToolInfo", "list_tools"]