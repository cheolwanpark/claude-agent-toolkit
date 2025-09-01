#!/usr/bin/env python3
# base.py - Simplified BaseTool class

from typing import Any, Optional

from .server import MCPServer
from .state_manager import StateManager
from ..logging import get_logger
from ..exceptions import StateError

logger = get_logger('tool')


class BaseTool:
    """
    Base class for MCP tools with state management and HTTP server support.
    
    Usage:
        class MyTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.state = {"my_data": []}
            
            @tool(description="My tool function")
            async def my_method(self, param: str) -> dict:
                # Tool logic here
                return {"result": "success"}
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the tool with state management."""
        # Store init args for potential serialization
        self._init_args = args
        self._init_kwargs = kwargs
        
        # Initialize state management
        self._state_manager = StateManager()
        
        # Server management
        self._server: Optional[MCPServer] = None
        self._host: str = "127.0.0.1"
        self._port: Optional[int] = None
    
    @property
    def state(self) -> Any:
        """Get current state."""
        return self._state_manager.state
    
    @state.setter
    def state(self, value: Any):
        """Set state value."""
        self._state_manager.state = value
        self._state_manager.version = 0  # Reset version on direct assignment
    
    @property
    def connection_url(self) -> str:
        """Get MCP connection URL."""
        if not self._port:
            raise StateError(
                "Tool is not running. Call tool.run() first, then access connection_url."
            )
        return f"http://{self._host}:{self._port}/mcp"  # no trailing slash
    
    @property
    def health_url(self) -> str:
        """Get health check URL."""
        if not self._port:
            raise StateError(
                "Tool is not running. Call tool.run() first, then access health_url."
            )
        return f"http://{self._host}:{self._port}/health"
    
    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, *, workers: Optional[int] = None, log_level: str = "ERROR") -> 'BaseTool':
        """
        Start the MCP server.
        
        Args:
            host: Host to bind to
            port: Port to bind to (auto-select if None)  
            workers: Number of worker processes (for CPU-bound operations)
            log_level: Logging level for FastMCP (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns:
            Self for chaining
        """
        if self._server:
            raise StateError("Already running")
        
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