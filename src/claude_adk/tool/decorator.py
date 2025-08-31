#!/usr/bin/env python3
# decorator.py - MCP tool method decorator

from typing import Callable, List, Optional


def tool(
    name: Optional[str] = None,
    description: str = "",
    *,
    cpu_bound: bool = False,
    timeout_s: int = 60,
    snapshot: Optional[List[str]] = None,
    conflict_policy: str = "retry",       # "retry" | "error"
    max_retries: int = 16,                # transparent retries on conflict
    backoff_initial_ms: int = 5,          # exp backoff start
    backoff_max_ms: int = 250,            # cap
):
    """
    Decorator to mark methods as MCP tools.
    
    Args:
        name: Tool name (defaults to function name)
        description: Tool description for MCP
        cpu_bound: Whether the tool is CPU-intensive
        timeout_s: Timeout in seconds for CPU-bound operations
        snapshot: List of config fields to snapshot for CPU-bound operations
        conflict_policy: How to handle state conflicts ("retry" or "error")
        max_retries: Maximum retry attempts for conflicts
        backoff_initial_ms: Initial backoff delay in milliseconds
        backoff_max_ms: Maximum backoff delay in milliseconds
    """
    def deco(fn: Callable):
        setattr(fn, "__mcp_tool__", True)
        setattr(fn, "__mcp_meta__", {
            "name": name or fn.__name__,
            "description": description,
            "cpu_bound": cpu_bound,
            "timeout_s": timeout_s,
            "snapshot": snapshot or [],
            "conflict_policy": conflict_policy,
            "max_retries": max_retries,
            "backoff_initial_ms": backoff_initial_ms,
            "backoff_max_ms": backoff_max_ms,
        })
        return fn
    return deco