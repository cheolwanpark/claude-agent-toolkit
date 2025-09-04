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
    The MCP server starts automatically when the tool is instantiated.
    
    Usage:
        # Basic usage (server cleaned up by __del__)
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
        
        # Context manager usage (recommended for explicit resource management)
        with MyTool(workers=2) as tool:
            agent = Agent(tools=[tool])
            result = await agent.run("Process data")
        # Server automatically cleaned up here
        
        # Multiple tools
        with MyTool() as calc_tool, WeatherTool() as weather_tool:
            agent = Agent(tools=[calc_tool, weather_tool])
            result = await agent.run("Calculate and check weather")
        # Both servers cleaned up automatically
    """
    
    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, *, workers: Optional[int] = None, log_level: str = "ERROR"):
        """
        Initialize the tool and automatically start the MCP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to (auto-select if None)  
            workers: Number of worker processes (for parallel operations)
            log_level: Logging level for FastMCP (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Server management only
        self._server: Optional[MCPServer] = None
        self._host: str = host
        self._port: Optional[int] = None
        
        # Auto-start server
        self._server = MCPServer(self, log_level=log_level)
        
        # Set worker count if specified
        if workers is not None:
            self._server.worker_manager.max_workers = max(1, int(workers))
        
        self._host, self._port = self._server.start(host, port)
        logger.info("%s @ %s", self.__class__.__name__, self.connection_url)
    
    @property
    def connection_url(self) -> str:
        """Get MCP connection URL."""
        return f"http://{self._host}:{self._port}/mcp"  # no trailing slash
    
    @property
    def health_url(self) -> str:
        """Get health check URL."""
        return f"http://{self._host}:{self._port}/health"
    
    def __enter__(self):
        """Enter context manager - tool server is already running from __init__."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - cleanup server resources."""
        if hasattr(self, '_server') and self._server:
            self._server.cleanup()
            self._server = None
            self._port = None
        return False  # Don't suppress exceptions
    
    def __del__(self):
        """Clean up server resources when tool is destroyed."""
        if hasattr(self, '_server') and self._server:
            self._server.cleanup()