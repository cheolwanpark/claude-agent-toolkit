#!/usr/bin/env python3
# base.py - Simplified BaseTool class

from typing import Optional

from .server import MCPServer
from ..logging import get_logger
from ..exceptions import ConnectionError

logger = get_logger('tool')


class BaseTool:
    """
    Base class for MCP tools with HTTP server support.
    
    Tools are stateless by design - manage your own data explicitly.
    
    Usage:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__()
                # Manage your own data explicitly
                self.my_data = []
            
            @tool(description="Async tool function")
            async def my_async_method(self, param: str) -> dict:
                # Async tool logic here
                return {"result": "success"}
            
            @tool(description="Parallel tool function", parallel=True)
            def my_parallel_method(self, param: str) -> dict:
                # Sync tool logic that runs in separate process
                return {"result": "success"}
    """
    
    def __init__(self):
        """Initialize the tool."""
        # Server management only
        self._server: Optional[MCPServer] = None
        self._host: str = "127.0.0.1"
        self._port: Optional[int] = None
    
    @property
    def connection_url(self) -> str:
        """Get MCP connection URL."""
        if not self._port:
            raise ConnectionError(
                "Tool is not running. Call tool.run() first, then access connection_url."
            )
        return f"http://{self._host}:{self._port}/mcp"  # no trailing slash
    
    @property
    def health_url(self) -> str:
        """Get health check URL."""
        if not self._port:
            raise ConnectionError(
                "Tool is not running. Call tool.run() first, then access health_url."
            )
        return f"http://{self._host}:{self._port}/health"
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, *, workers: Optional[int] = None, log_level: str = "ERROR") -> 'BaseTool':
        """
        Start the MCP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to (auto-select if None)  
            workers: Number of worker processes (for parallel operations)
            log_level: Logging level for FastMCP (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Self for chaining
        """
        if self._server:
            raise ConnectionError("Already running")
        
        self._server = MCPServer(self, log_level=log_level)
        
        # Set worker count if specified
        if workers is not None:
            self._server.worker_manager.max_workers = max(1, int(workers))
        
        self._host, self._port = self._server.start(host, port)
        logger.info("%s @ %s", self.__class__.__name__, self.connection_url)
        return self
    
    def cleanup(self):
        """Clean up server resources."""
        if self._server:
            self._server.cleanup()