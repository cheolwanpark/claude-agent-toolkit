#!/usr/bin/env python3
# decorator.py - MCP tool method decorator

import asyncio
from typing import Callable, Optional

from ..exceptions import ConfigurationError


def tool(
    name: Optional[str] = None,
    description: str = "",
    *,
    parallel: bool = False,
    timeout_s: int = 60,
):
    """
    Decorator to mark methods as MCP tools.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description for MCP
        parallel: Whether the tool runs in a separate process (must be sync function)
        timeout_s: Timeout in seconds for parallel operations
        
    Validation Rules:
        - parallel=True requires sync function (def, not async def)
        - parallel=False requires async function (async def)
    """
    def deco(fn: Callable):
        # Validate async/parallel combinations
        is_async = asyncio.iscoroutinefunction(fn)
        
        if parallel and is_async:
            raise ConfigurationError(
                f"Tool '{name or fn.__name__}' cannot use parallel=True with async functions. "
                "Use parallel=True with sync functions or parallel=False with async functions."
            )
        
        if not parallel and not is_async:
            raise ConfigurationError(
                f"Tool '{name or fn.__name__}' requires async function when parallel=False. "
                "Use 'async def' or set parallel=True for sync functions."
            )
        
        setattr(fn, "__mcp_tool__", True)
        setattr(fn, "__mcp_meta__", {
            "name": name or fn.__name__,
            "description": description,
            "parallel": parallel,
            "timeout_s": timeout_s,
        })
        return fn
    return deco