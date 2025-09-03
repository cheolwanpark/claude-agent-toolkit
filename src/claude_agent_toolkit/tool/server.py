#!/usr/bin/env python3
# server.py - HTTP server management for MCP tools

import inspect
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from fastmcp import FastMCP
from fastmcp.utilities.logging import configure_logging

from .worker import WorkerPoolManager
from ..exceptions import ConnectionError, ExecutionError


class MCPServer:
    """HTTP server for MCP tool endpoints."""
    
    def __init__(self, tool_instance: Any, log_level: str = "ERROR"):
        """
        Initialize MCP server for a tool instance.
        
        Args:
            tool_instance: Tool instance to serve
            log_level: Logging level for FastMCP (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.tool_instance = tool_instance
        self.worker_manager = WorkerPoolManager()
        self._server_thread: Optional[threading.Thread] = None
        self._ready = False
        self._log_level = log_level
        
        # Configure FastMCP logging
        configure_logging(level=log_level)
    
    def _pick_port(self, host: str) -> int:
        """Pick an available port on the given host."""
        try:
            s = socket.socket()
            s.bind((host, 0))
            port = s.getsockname()[1]
            s.close()
            return port
        except socket.error as e:
            raise ConnectionError(f"Failed to bind to host {host}: {e}") from e
        except Exception as e:
            raise ConnectionError(f"Socket operation failed: {e}") from e
    
    def _collect_tool_methods(self) -> List[Tuple[Callable, Dict[str, Any]]]:
        """Collect all methods marked with @tool decorator."""
        methods = []
        
        for name in dir(self.tool_instance):
            # Skip private/magic methods
            if name.startswith('_'):
                continue
                
            # Use getattr_static to check if it's a property without evaluation
            static_attr = inspect.getattr_static(type(self.tool_instance), name, None)
            if isinstance(static_attr, property):
                continue  # Skip all properties - we only want methods
                
            try:
                member = getattr(self.tool_instance, name)
                if inspect.ismethod(member) and getattr(member, "__mcp_tool__", False):
                    meta = getattr(member, "__mcp_meta__", {})
                    methods.append((member, meta))
            except (AttributeError, TypeError):
                # Skip attributes that can't be accessed or aren't callable
                continue
        
        return methods
    
    def _register_parallel_tool(self, mcp: FastMCP, method: Callable, meta: Dict[str, Any]):
        """Register parallel tool method with worker pool execution."""
        sig = inspect.signature(method)
        params = [str(p) for p in sig.parameters.values()]
        param_list = ", ".join(params)
        param_names = [p.name for p in sig.parameters.values()]

        async def __dispatcher__(**kw):
            args = tuple(kw[p] for p in param_names)
            
            try:
                result = await self.worker_manager.execute_parallel(method, meta, args, {})
                return result
            except ExecutionError:
                # Re-raise ExecutionError as-is
                raise
            except Exception as e:
                raise ExecutionError(f"Parallel operation failed: {e}") from e

        # Generate wrapper with SAME signature as original
        ns: Dict[str, Any] = {"__dispatcher__": __dispatcher__}
        code = f"async def _wrapped({param_list}):\n    kw = dict()\n"
        for name in param_names:
            code += f"    kw['{name}'] = {name}\n"
        code += "    return await __dispatcher__(**kw)\n"
        exec(code, ns)
        wrapped = ns["_wrapped"]
        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        try:
            wrapped.__annotations__ = method.__func__.__annotations__
        except AttributeError:
            # Method might not have annotations or __func__ attribute
            pass

        mcp.tool(name=meta["name"], description=meta["description"])(wrapped)
    
    def _create_mcp_app(self) -> FastMCP:
        """Create FastMCP application with tool endpoints."""
        mcp = FastMCP(name=self.tool_instance.__class__.__name__)

        @mcp.custom_route("/health", methods=["GET"])
        async def health(_):
            from starlette.responses import PlainTextResponse
            return PlainTextResponse("OK", status_code=200)

        # Register all tool methods
        for method, meta in self._collect_tool_methods():
            if meta.get("parallel", False):
                self._register_parallel_tool(mcp, method, meta)
            else:
                mcp.tool(name=meta["name"], description=meta["description"])(method)

        return mcp
    
    def _run_server_thread(self, host: str, port: int):
        """Run MCP server in a separate thread."""
        mcp = self._create_mcp_app()
        mcp.run(
            transport="http", 
            host=host, 
            port=port,
            show_banner=False,  # Always suppress banner
            log_level=self._log_level
        )
    
    def start(self, host: str = "127.0.0.1", port: Optional[int] = None) -> tuple[str, int]:
        """
        Start the MCP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to (auto-select if None)
            
        Returns:
            Tuple of (host, port) the server is running on
        """
        if self._server_thread:
            raise ConnectionError("Server already running")
        
        actual_port = port or self._pick_port(host)
        
        self._server_thread = threading.Thread(
            target=self._run_server_thread,
            args=(host, actual_port),
            daemon=True,
        )
        self._server_thread.start()
        self._wait_ready(host, actual_port)
        return host, actual_port
    
    def _wait_ready(self, host: str, port: int, timeout: float = 10.0):
        """Wait for server to be ready."""
        health_url = f"http://{host}:{port}/health"
        start = time.time()
        last_err = None
        
        while time.time() - start < timeout:
            try:
                r = httpx.get(health_url, timeout=0.5)
                if r.status_code == 200:
                    self._ready = True
                    return
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_err = e
            except Exception as e:
                last_err = e
            time.sleep(0.1)
        
        raise ConnectionError(f"MCP server not ready at {health_url} after {timeout}s (last error: {last_err})")
    
    def cleanup(self):
        """Clean up server resources."""
        self.worker_manager.cleanup()