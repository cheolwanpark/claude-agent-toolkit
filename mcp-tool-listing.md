# MCP Tool Listing Guide

Comprehensive guide for extracting and listing available tools from Model Context Protocol (MCP) servers using various transport methods and programming approaches.

## Table of Contents

- [Overview](#overview)
- [Transport Methods](#transport-methods)
- [Python SDK Examples](#python-sdk-examples)
- [Claude Agent Toolkit Integration](#claude-agent-toolkit-integration)
- [Command Line Tools](#command-line-tools)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The Model Context Protocol (MCP) provides standardized methods for discovering and interacting with tools through the `list_tools` command. This command works across all transport methods and returns tool metadata including names, descriptions, and parameter schemas.

### Key Concepts

- **Transport Agnostic**: `list_tools()` works with STDIO, HTTP+SSE, and other transport methods
- **JSON-RPC 2.0**: Underlying protocol for all MCP communications
- **Caching**: Many implementations cache tool lists for performance
- **Discovery**: Tools are discovered dynamically at runtime

## Transport Methods

### STDIO Transport

STDIO servers run as subprocesses and communicate via standard input/output streams.

**Characteristics:**
- Local execution environment
- Direct process communication
- Suitable for trusted, local tools
- Automatic lifecycle management

### HTTP with Server-Sent Events (SSE)

HTTP+SSE servers run remotely and are accessed via URLs.

**Characteristics:**
- Remote server execution
- HTTP for client requests
- Server-Sent Events for streaming responses
- Suitable for distributed architectures

### Streamable HTTP

Modern, recommended approach for remote MCP servers.

**Characteristics:**
- Uses streamable HTTP transport
- JSON-RPC 2.0 over HTTP
- Enhanced performance and reliability
- Official MCP specification compliance

## Python SDK Examples

### Using MCP Python SDK - STDIO Transport

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_tools_stdio():
    """List tools from an MCP server using STDIO transport."""
    server_params = StdioServerParameters(
        command="python", 
        args=["path/to/your/mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP session
            await session.initialize()
            
            # Call list_tools to get available tools
            result = await session.list_tools()
            
            print("Available tools:")
            for tool in result.tools:
                print(f"- {tool.name}: {tool.description}")
                print(f"  Input Schema: {tool.inputSchema}")
                if hasattr(tool, 'parameters'):
                    print(f"  Parameters: {tool.parameters}")

if __name__ == "__main__":
    asyncio.run(list_tools_stdio())
```

### Using MCP Python SDK - HTTP Transport

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def list_tools_http():
    """List tools from an MCP server using HTTP+SSE transport."""
    server_url = "http://localhost:8080/sse"
    
    try:
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List available tools from HTTP server
                result = await session.list_tools()
                
                print(f"Connected to: {server_url}")
                print("Available tools:")
                for tool in result.tools:
                    print(f"  Tool: {tool.name}")
                    print(f"  Description: {tool.description}")
                    print(f"  Input Schema: {tool.inputSchema}")
                    print("  ---")
                    
    except Exception as e:
        print(f"Error connecting to {server_url}: {e}")

if __name__ == "__main__":
    asyncio.run(list_tools_http())
```

### Advanced MCP Client with Error Handling

```python
import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolDiscovery:
    """Advanced MCP client for tool discovery across transport methods."""
    
    async def discover_stdio_tools(self, command: List[str]) -> List[Dict[str, Any]]:
        """Discover tools from STDIO MCP server."""
        server_params = StdioServerParameters(
            command=command[0],
            args=command[1:] if len(command) > 1 else []
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                            "transport": "stdio"
                        }
                        for tool in result.tools
                    ]
        except Exception as e:
            logger.error(f"Failed to discover STDIO tools: {e}")
            return []
    
    async def discover_http_tools(self, url: str) -> List[Dict[str, Any]]:
        """Discover tools from HTTP+SSE MCP server."""
        try:
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                            "transport": "http",
                            "server_url": url
                        }
                        for tool in result.tools
                    ]
        except Exception as e:
            logger.error(f"Failed to discover HTTP tools from {url}: {e}")
            return []
    
    async def discover_all_tools(self, 
                               stdio_commands: List[List[str]] = None,
                               http_urls: List[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Discover tools from multiple MCP servers."""
        results = {"stdio_tools": [], "http_tools": []}
        
        # Discover STDIO tools
        if stdio_commands:
            for command in stdio_commands:
                tools = await self.discover_stdio_tools(command)
                results["stdio_tools"].extend(tools)
        
        # Discover HTTP tools
        if http_urls:
            for url in http_urls:
                tools = await self.discover_http_tools(url)
                results["http_tools"].extend(tools)
        
        return results

# Usage example
async def main():
    discovery = MCPToolDiscovery()
    
    # Define servers to discover
    stdio_servers = [
        ["python", "examples/calculator/server.py"],
        ["python", "examples/filesystem/server.py"]
    ]
    
    http_servers = [
        "http://localhost:8080/sse",
        "http://localhost:8081/sse"
    ]
    
    # Discover all tools
    all_tools = await discovery.discover_all_tools(
        stdio_commands=stdio_servers,
        http_urls=http_servers
    )
    
    # Print results
    print("=== MCP Tool Discovery Results ===")
    for transport_type, tools in all_tools.items():
        print(f"\n{transport_type.upper()}:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Claude Agent Toolkit Integration

### STDIO Tool Connector

```python
import asyncio
from claude_agent_toolkit.agent.tool_connector import StdioToolConnector

async def get_tools_stdio():
    """Get tools using Claude Agent Toolkit STDIO connector."""
    command = ["python", "src/examples/calculator/tool.py"]
    connector = StdioToolConnector(command=command)
    
    try:
        print(f"Starting tool server: {' '.join(command)}")
        tools = await connector.list_tools()
        
        print("\n--- Available Tools (STDIO) ---")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            if hasattr(tool, 'parameters'):
                params = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
                print(f"  Parameters: {params}")
        print("--------------------------------")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await connector.close()
        print("STDIO connector closed.")

asyncio.run(get_tools_stdio())
```

### HTTP Tool Connector

```python
import asyncio
from claude_agent_toolkit.agent.tool_connector import HttpToolConnector

async def get_tools_http():
    """Get tools using Claude Agent Toolkit HTTP connector."""
    url = "http://127.0.0.1:8080"
    connector = HttpToolConnector(url=url)
    
    try:
        print(f"Connecting to: {url}")
        tools = await connector.list_tools()
        
        print("\n--- Available Tools (HTTP) ---")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            if hasattr(tool, 'input_schema'):
                print(f"  Schema: {tool.input_schema}")
        print("-------------------------------")
        
    except Exception as e:
        print(f"Connection error: {e}")
        print("Make sure the HTTP server is running at the specified URL")
    finally:
        await connector.close()
        print("HTTP connector closed.")

# Before running this, start your HTTP server:
# python src/examples/calculator/tool.py --host 127.0.0.1 --port 8080
asyncio.run(get_tools_http())
```

### Agent Integration Pattern

```python
import asyncio
from claude_agent_toolkit import Agent, BaseTool, tool
from claude_agent_toolkit.agent.tool_connector import HttpToolConnector, StdioToolConnector

class ToolDiscoveryAgent:
    """Agent that can discover and list tools from various MCP servers."""
    
    def __init__(self):
        self.discovered_tools = {}
    
    async def discover_tools_from_configs(self, server_configs: list):
        """Discover tools from multiple server configurations."""
        for config in server_configs:
            transport_type = config.get("transport", "stdio")
            name = config.get("name", f"server_{len(self.discovered_tools)}")
            
            if transport_type == "stdio":
                connector = StdioToolConnector(command=config["command"])
            elif transport_type == "http":
                connector = HttpToolConnector(url=config["url"])
            else:
                continue
            
            try:
                tools = await connector.list_tools()
                self.discovered_tools[name] = {
                    "transport": transport_type,
                    "config": config,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "schema": getattr(tool, 'input_schema', None) or getattr(tool, 'inputSchema', None)
                        }
                        for tool in tools
                    ]
                }
            except Exception as e:
                print(f"Failed to discover tools from {name}: {e}")
            finally:
                await connector.close()
    
    def print_discovery_report(self):
        """Print a comprehensive report of discovered tools."""
        print("=== MCP Tool Discovery Report ===")
        
        total_tools = sum(len(server["tools"]) for server in self.discovered_tools.values())
        print(f"Total servers: {len(self.discovered_tools)}")
        print(f"Total tools: {total_tools}")
        
        for server_name, server_info in self.discovered_tools.items():
            print(f"\n[{server_name.upper()}] ({server_info['transport']})")
            for tool in server_info["tools"]:
                print(f"  ✓ {tool['name']}: {tool['description']}")
        
        print("\n" + "="*35)

# Usage example
async def main():
    agent = ToolDiscoveryAgent()
    
    # Define server configurations
    server_configs = [
        {
            "name": "calculator",
            "transport": "stdio",
            "command": ["python", "src/examples/calculator/tool.py"]
        },
        {
            "name": "weather",
            "transport": "http",
            "url": "http://localhost:8081/sse"
        }
    ]
    
    # Discover tools
    await agent.discover_tools_from_configs(server_configs)
    
    # Print report
    agent.print_discovery_report()

if __name__ == "__main__":
    asyncio.run(main())
```

## Command Line Tools

### Using mcptools CLI

Install and use the `mcptools` command-line interface:

```bash
# Install mcptools
pip install mcptools

# List tools from HTTP server
mcptools http://localhost:8080 list-tools

# List tools from STDIO server  
mcptools stdio python server.py list-tools

# Interactive mode
mcptools http://localhost:8080 --interactive

# With custom headers for authentication
mcptools http://localhost:8080 list-tools \
  --header "Authorization: Bearer your-token"
```

### Custom CLI Tool Discovery Script

```python
#!/usr/bin/env python3
"""
Custom CLI tool for MCP server discovery.
Usage: python mcp-discover.py [config.json]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

async def discover_stdio_server(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover tools from STDIO server."""
    server_params = StdioServerParameters(
        command=config["command"][0],
        args=config["command"][1:] if len(config["command"]) > 1 else []
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": tool.name, "description": tool.description} for tool in result.tools]

async def discover_http_server(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover tools from HTTP+SSE server."""
    async with sse_client(config["url"]) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [{"name": tool.name, "description": tool.description} for tool in result.tools]

async def main():
    parser = argparse.ArgumentParser(description="Discover MCP server tools")
    parser.add_argument("config", help="Configuration file (JSON)")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--format", choices=["json", "yaml", "table"], default="table")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    
    # Discover tools from all configured servers
    results = {}
    
    for server_name, server_config in config.get("servers", {}).items():
        try:
            if server_config["transport"] == "stdio":
                tools = await discover_stdio_server(server_config)
            elif server_config["transport"] == "http":
                tools = await discover_http_server(server_config)
            else:
                print(f"Unknown transport: {server_config['transport']}")
                continue
            
            results[server_name] = {
                "transport": server_config["transport"],
                "tools": tools
            }
            
        except Exception as e:
            print(f"Failed to discover tools from {server_name}: {e}")
    
    # Output results
    if args.format == "json":
        output = json.dumps(results, indent=2)
    elif args.format == "table":
        output = format_table_output(results)
    else:
        output = str(results)  # Fallback
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)

def format_table_output(results: Dict[str, Any]) -> str:
    """Format results as a table."""
    lines = ["MCP Tool Discovery Results", "=" * 30, ""]
    
    for server_name, server_data in results.items():
        lines.append(f"Server: {server_name} ({server_data['transport']})")
        lines.append("-" * 40)
        
        for tool in server_data["tools"]:
            lines.append(f"  ✓ {tool['name']}: {tool['description']}")
        
        lines.append("")
    
    return "\n".join(lines)

if __name__ == "__main__":
    asyncio.run(main())
```

Example configuration file (`mcp-config.json`):

```json
{
  "servers": {
    "calculator": {
      "transport": "stdio",
      "command": ["python", "examples/calculator/server.py"]
    },
    "filesystem": {
      "transport": "http",
      "url": "http://localhost:8080/sse"
    },
    "weather": {
      "transport": "stdio",
      "command": ["node", "weather-server.js"]
    }
  }
}
```

## Best Practices

### 1. Error Handling and Resilience

```python
import asyncio
import logging
from typing import Optional, List, Dict, Any

async def robust_tool_discovery(server_configs: List[Dict[str, Any]], 
                              timeout: float = 30.0) -> Dict[str, Any]:
    """Robust tool discovery with error handling and timeouts."""
    results = {"successful": {}, "failed": {}}
    
    for config in server_configs:
        server_name = config.get("name", "unknown")
        
        try:
            # Use asyncio.wait_for to enforce timeout
            tools = await asyncio.wait_for(
                discover_tools_from_config(config),
                timeout=timeout
            )
            
            results["successful"][server_name] = tools
            
        except asyncio.TimeoutError:
            results["failed"][server_name] = "Timeout after {timeout}s"
            logging.error(f"Tool discovery timeout for {server_name}")
            
        except Exception as e:
            results["failed"][server_name] = str(e)
            logging.error(f"Tool discovery failed for {server_name}: {e}")
    
    return results

async def discover_tools_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover tools based on configuration."""
    transport = config.get("transport", "stdio")
    
    if transport == "stdio":
        return await discover_stdio_tools(config["command"])
    elif transport == "http":
        return await discover_http_tools(config["url"])
    else:
        raise ValueError(f"Unsupported transport: {transport}")
```

### 2. Caching and Performance

```python
import asyncio
import time
from functools import wraps
from typing import Dict, Any, Tuple

class ToolDiscoveryCache:
    """Cache for MCP tool discovery results."""
    
    def __init__(self, ttl: float = 300):  # 5 minutes default TTL
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if still valid."""
        if key in self.cache:
            timestamp, result = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Cache a result."""
        self.cache[key] = (time.time(), value)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()

# Global cache instance
discovery_cache = ToolDiscoveryCache(ttl=300)

def cached_discovery(func):
    """Decorator for caching tool discovery results."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Create cache key from arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        
        # Try to get from cache
        cached_result = discovery_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function and cache result
        result = await func(*args, **kwargs)
        discovery_cache.set(cache_key, result)
        return result
    
    return wrapper

@cached_discovery
async def discover_tools_cached(server_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cached version of tool discovery."""
    return await discover_tools_from_config(server_config)
```

### 3. Parallel Discovery

```python
import asyncio
from typing import List, Dict, Any

async def parallel_tool_discovery(server_configs: List[Dict[str, Any]], 
                                max_concurrent: int = 5) -> Dict[str, Any]:
    """Discover tools from multiple servers in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def discover_single_server(config: Dict[str, Any]) -> Tuple[str, Any]:
        """Discover tools from a single server with concurrency control."""
        async with semaphore:
            server_name = config.get("name", "unknown")
            try:
                tools = await discover_tools_from_config(config)
                return server_name, {"status": "success", "tools": tools}
            except Exception as e:
                return server_name, {"status": "error", "error": str(e)}
    
    # Run all discoveries in parallel
    tasks = [discover_single_server(config) for config in server_configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    discovery_results = {}
    for result in results:
        if isinstance(result, Exception):
            # Handle exceptions from gather
            continue
        
        server_name, server_result = result
        discovery_results[server_name] = server_result
    
    return discovery_results
```

### 4. Configuration Management

```python
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class MCPConfig:
    """Configuration manager for MCP tool discovery."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "mcp-config.json",
            os.path.expanduser("~/.mcp/config.json"),
            "/etc/mcp/config.json"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("No MCP configuration file found")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
    
    def get_server_configs(self) -> List[Dict[str, Any]]:
        """Get list of server configurations."""
        servers = []
        for name, config in self.config_data.get("servers", {}).items():
            server_config = config.copy()
            server_config["name"] = name
            servers.append(server_config)
        return servers
    
    def get_server_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific server."""
        server_config = self.config_data.get("servers", {}).get(name)
        if server_config:
            config = server_config.copy()
            config["name"] = name
            return config
        return None
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if "servers" not in self.config_data:
            errors.append("Missing 'servers' section in configuration")
            return errors
        
        for name, config in self.config_data["servers"].items():
            if "transport" not in config:
                errors.append(f"Server '{name}': Missing 'transport' field")
            elif config["transport"] == "stdio" and "command" not in config:
                errors.append(f"Server '{name}': STDIO transport requires 'command' field")
            elif config["transport"] == "http" and "url" not in config:
                errors.append(f"Server '{name}': HTTP transport requires 'url' field")
        
        return errors

# Usage example
config = MCPConfig("my-mcp-config.json")
errors = config.validate_config()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
else:
    server_configs = config.get_server_configs()
    results = await parallel_tool_discovery(server_configs)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Connection Timeouts

```python
# Issue: Server takes too long to respond
# Solution: Increase timeout and add retry logic

import asyncio

async def discovery_with_retry(config: Dict[str, Any], 
                             max_retries: int = 3,
                             timeout: float = 30.0) -> Optional[List[Dict[str, Any]]]:
    """Tool discovery with retry logic."""
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                discover_tools_from_config(config),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            raise
    
    return None
```

#### 2. Server Startup Issues

```python
# Issue: STDIO server fails to start
# Solution: Validate command and environment

import subprocess
import shutil
from pathlib import Path

def validate_stdio_command(command: List[str]) -> List[str]:
    """Validate STDIO command before use."""
    issues = []
    
    # Check if executable exists
    executable = command[0]
    if not shutil.which(executable):
        issues.append(f"Executable '{executable}' not found in PATH")
    
    # Check if script file exists (for Python, Node.js, etc.)
    if len(command) > 1:
        script_path = Path(command[1])
        if not script_path.exists():
            issues.append(f"Script file '{script_path}' does not exist")
        elif not script_path.is_file():
            issues.append(f"Script path '{script_path}' is not a file")
    
    return issues

# Example usage
command = ["python", "nonexistent_server.py"]
issues = validate_stdio_command(command)
if issues:
    print("Command validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

#### 3. HTTP Server Connectivity

```python
# Issue: Cannot connect to HTTP server
# Solution: Health check and connection validation

import aiohttp
import asyncio

async def validate_http_server(url: str) -> Dict[str, Any]:
    """Validate HTTP server connectivity."""
    result = {"reachable": False, "mcp_compatible": False, "error": None}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Basic connectivity check
            async with session.get(url, timeout=10) as response:
                result["reachable"] = True
                result["status_code"] = response.status
                
                # Check for MCP compatibility (simplified)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    result["mcp_compatible"] = True
                
    except aiohttp.ClientConnectorError as e:
        result["error"] = f"Connection error: {e}"
    except asyncio.TimeoutError:
        result["error"] = "Connection timeout"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
    
    return result

# Example usage
url = "http://localhost:8080/sse"
validation_result = await validate_http_server(url)
if not validation_result["reachable"]:
    print(f"Server at {url} is not reachable: {validation_result['error']}")
```

#### 4. Debugging Tool Discovery

```python
# Enable detailed logging for debugging

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable MCP-specific logging
logger = logging.getLogger("mcp")
logger.setLevel(logging.DEBUG)

# Add debug wrapper for tool discovery
async def debug_tool_discovery(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tool discovery with detailed debugging."""
    server_name = config.get("name", "unknown")
    transport = config.get("transport", "unknown")
    
    logger.info(f"Starting tool discovery for {server_name} ({transport})")
    
    try:
        start_time = asyncio.get_event_loop().time()
        tools = await discover_tools_from_config(config)
        end_time = asyncio.get_event_loop().time()
        
        logger.info(f"Discovery completed for {server_name} in {end_time - start_time:.2f}s")
        logger.debug(f"Found {len(tools)} tools: {[t['name'] for t in tools]}")
        
        return tools
        
    except Exception as e:
        logger.error(f"Discovery failed for {server_name}: {e}", exc_info=True)
        raise
```

### Performance Monitoring

```python
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DiscoveryMetrics:
    """Metrics for tool discovery performance."""
    server_name: str
    transport: str
    start_time: float
    end_time: float
    tool_count: int
    success: bool
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

class PerformanceMonitor:
    """Monitor performance of tool discovery operations."""
    
    def __init__(self):
        self.metrics: List[DiscoveryMetrics] = []
    
    async def monitored_discovery(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform tool discovery with performance monitoring."""
        server_name = config.get("name", "unknown")
        transport = config.get("transport", "unknown")
        start_time = time.time()
        
        try:
            tools = await discover_tools_from_config(config)
            end_time = time.time()
            
            metric = DiscoveryMetrics(
                server_name=server_name,
                transport=transport,
                start_time=start_time,
                end_time=end_time,
                tool_count=len(tools),
                success=True
            )
            self.metrics.append(metric)
            return tools
            
        except Exception as e:
            end_time = time.time()
            
            metric = DiscoveryMetrics(
                server_name=server_name,
                transport=transport,
                start_time=start_time,
                end_time=end_time,
                tool_count=0,
                success=False,
                error=str(e)
            )
            self.metrics.append(metric)
            raise
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        successful = [m for m in self.metrics if m.success]
        failed = [m for m in self.metrics if not m.success]
        
        report = {
            "total_discoveries": len(self.metrics),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.metrics) * 100,
            "avg_duration": sum(m.duration for m in successful) / len(successful) if successful else 0,
            "total_tools": sum(m.tool_count for m in successful),
            "by_transport": {}
        }
        
        # Group by transport
        transports = {}
        for metric in self.metrics:
            if metric.transport not in transports:
                transports[metric.transport] = []
            transports[metric.transport].append(metric)
        
        for transport, metrics in transports.items():
            successful_transport = [m for m in metrics if m.success]
            report["by_transport"][transport] = {
                "total": len(metrics),
                "successful": len(successful_transport),
                "avg_duration": sum(m.duration for m in successful_transport) / len(successful_transport) if successful_transport else 0
            }
        
        return report

# Example usage
monitor = PerformanceMonitor()

server_configs = [
    {"name": "calc", "transport": "stdio", "command": ["python", "calc.py"]},
    {"name": "web", "transport": "http", "url": "http://localhost:8080/sse"}
]

for config in server_configs:
    try:
        tools = await monitor.monitored_discovery(config)
        print(f"Discovered {len(tools)} tools from {config['name']}")
    except Exception as e:
        print(f"Failed to discover tools from {config['name']}: {e}")

# Print performance report
report = monitor.get_performance_report()
print(json.dumps(report, indent=2))
```

---

This comprehensive guide covers all major aspects of MCP tool listing across different transport methods, programming approaches, and practical use cases. The examples provide production-ready code patterns that can be adapted for specific requirements.