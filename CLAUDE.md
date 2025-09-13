# CLAUDE.md

Production-ready Python framework for building Claude Code agents with custom MCP tools.

## Overview & Features
- **Production Architecture**: Enterprise-grade framework with explicit data management
- **MCP Tools**: HTTP-based Model Context Protocol servers with auto-discovery
- **Dual Execution**: Docker isolation (production) or subprocess (development)
- **Parallel Processing**: CPU-intensive operations in separate worker processes
- **No Hidden State**: Users control their own data explicitly

## Architecture

Custom MCP tools run as HTTP servers on host machine. Claude Code executes in Docker container or direct subprocess.

### Core Components
- **Agent** (`core.py`): Main orchestrator managing agent lifecycle
- **Executors**: DockerExecutor (production) or SubprocessExecutor (development)
- **ToolConnector**: Manages MCP tool server connections
- **AbstractTool**: Base interface for all MCP tools (internal and external)
- **BaseTool**: HTTP-based MCP tool class for custom tools with @tool decorator
- **MCPTool Classes**: `StdioMCPTool`, `HttpMCPTool` for external MCP server integration
- **MCPServer**: FastMCP HTTP server with auto port selection
- **Built-in Tools**: FileSystemTool, DataTransferTool

### Executors

**DockerExecutor (Default - Production)**:
- Pre-built image: `cheolwanpark/claude-agent-toolkit:0.2.2`
- Full isolation with host networking for MCP access
- Automatic version matching between package and Docker image

**SubprocessExecutor (Development)**:
- Direct `claude-code-sdk` execution, no Docker dependency
- 6x faster startup (~0.5s vs ~3s)
- Temporary directory isolation

## Quick Start

### Basic Tool Pattern
```python
from claude_agent_toolkit import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.data = {}  # Explicit data management

    @tool(description="Async operation")
    async def process_async(self, data: str) -> dict:
        return {"result": f"processed_{data}"}

    @tool(description="Parallel operation", parallel=True, timeout_s=120)
    def process_parallel(self, data: str) -> dict:
        # Sync function - runs in separate process
        return {"result": f"heavy_{data}"}
```

### Agent Usage
```python
from claude_agent_toolkit import Agent, ExecutorType
from claude_agent_toolkit.tool.mcp import StdioMCPTool, HttpMCPTool

# Docker executor (default)
agent = Agent(tools=[MyTool()])

# Subprocess executor (faster startup)
agent = Agent(tools=[MyTool()], executor=ExecutorType.SUBPROCESS)

# External MCP server integration
stdio_tool = StdioMCPTool(command="node", args=["server.js"], name="my-server")
http_tool = HttpMCPTool(url="http://localhost:8080/mcp", name="http-server")
agent = Agent(tools=[MyTool(), stdio_tool, http_tool])

result = await agent.run("Process my data")
```

## Setup

```bash
export CLAUDE_CODE_OAUTH_TOKEN='your-token'
uv sync

# Run examples
cd src/examples/calculator && python main.py
cd src/examples/weather && python main.py
cd src/examples/subprocess && python main.py  # No Docker needed
```

## Configuration & Dependencies

### Key Dependencies
- Python 3.12+, `uv` package manager
- Docker Desktop (for DockerExecutor)
- Required: `docker`, `fastmcp`, `httpx`, `jsonpatch`, `uvicorn`
- Claude Code OAuth token from [claude.ai/code](https://claude.ai/code)

### Model Selection
```python
# Available models
agent = Agent(model="haiku")   # Fast, low cost
agent = Agent(model="sonnet")  # Balanced (default)
agent = Agent(model="opus")    # Most capable

# Override per run
result = await agent.run("Simple task", model="haiku")
result = await agent.run("Complex analysis", model="opus")
```

## Release Commands
```bash
# Test release
git tag v0.1.2b1 && git push origin v0.1.2b1

# Official release
git tag v0.1.2 && git push origin v0.1.2
```

## External MCP Server Integration

### Tool Architecture
The framework now supports both internal and external MCP tools through a unified `AbstractTool` interface:
- **BaseTool**: HTTP-based tools you create with @tool decorator
- **StdioMCPTool**: Connect to external MCP servers via stdin/stdout
- **HttpMCPTool**: Connect to external MCP servers via HTTP

### External Tool Classes
```python
from claude_agent_toolkit.tool.mcp import StdioMCPTool, HttpMCPTool

# Stdio transport (most common)
stdio_tool = StdioMCPTool(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
    name="everything-server"
)

# HTTP transport
http_tool = HttpMCPTool(
    url="http://localhost:3001/mcp",
    name="my-http-server"
)

# Use with agent like any other tool
agent = Agent(tools=[stdio_tool, http_tool])
```

### Transport Support
- **Stdio**: Direct command execution with stdin/stdout communication
- **HTTP**: REST-style HTTP endpoints for MCP protocol
- **Mixed Mode**: Combine internal BaseTool instances with external MCP servers

## Key Concepts

### Data Management
- **Explicit Control**: Users manage data as instance variables
- **No Hidden State**: No automatic versioning or conflict resolution
- **Parallel Operations**: `parallel=True` tools run in separate processes with new instances

### Tool Development Rules
- **Async functions**: Use `@tool()` decorator (default parallel=False)
- **Sync functions**: Use `@tool(parallel=True)` for CPU-intensive operations
- **Error Handling**: Wrap exceptions in framework exception types
- **State Management**: Use semaphores/atomic types for shared data in parallel tools

### Version Safety
- Docker image version automatically matches installed package version
- No fallback policy - exact version match required for maximum safety

## API Reference

### Agent Class
```python
Agent(
    oauth_token: str = None,  # Or use CLAUDE_CODE_OAUTH_TOKEN env var
    system_prompt: str = None,
    tools: List[BaseTool] = None,
    model: str = "sonnet",  # "haiku", "sonnet", "opus"
    executor: ExecutorType = ExecutorType.DOCKER
)

# Methods
agent.connect(tool: BaseTool)  # Add tool after initialization
await agent.run(prompt: str, verbose: bool = False, model: str = None) -> str
```

### @tool Decorator
```python
@tool(
    name: str = None,        # Defaults to function name
    description: str = "",   # Required for Claude Code
    parallel: bool = False,  # True = sync function, False = async function
    timeout_s: int = 60      # Timeout for parallel operations
)
```

### External MCP Tool Classes
```python
from claude_agent_toolkit.tool.mcp import StdioMCPTool, HttpMCPTool

# Stdio MCP Tool
StdioMCPTool(
    command: str,                    # Command to execute (e.g., "node", "python")
    args: List[str] = None,          # Arguments for the command
    env: Dict[str, str] = None,      # Environment variables
    name: str = None                 # Tool identifier (defaults to command name)
)

# HTTP MCP Tool
HttpMCPTool(
    url: str,                        # HTTP endpoint URL
    name: str = None                 # Tool identifier (defaults to hostname)
)
```

### Exception Hierarchy
```python
ClaudeAgentError           # Base exception
├── ConfigurationError     # Missing tokens, invalid config
├── ConnectionError        # Docker, network, port issues
└── ExecutionError         # Tool failures, timeouts
```

### Logging
```python
from claude_agent_toolkit import set_logging, LogLevel

set_logging(LogLevel.INFO)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)
```

## Common Issues

### ConfigurationError
- **Missing OAuth token**: Set `CLAUDE_CODE_OAUTH_TOKEN` environment variable
- **Invalid tool config**: Check tool initialization parameters

### ConnectionError
- **Docker not running**: Start Docker Desktop (`docker --version` to verify)
- **Port conflicts**: Tools auto-select available ports, check with `docker ps`
- **Network issues**: Verify Docker daemon is accessible

### ExecutionError
- **Tool timeouts**: Increase `timeout_s` for parallel operations
- **Implementation errors**: Check tool method implementations and exception handling

### Debug Mode
```python
# Enable debug logging
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Verbose execution
result = await agent.run("prompt", verbose=True)

# Check tool health: visit http://localhost:{port}/health
```

## Best Practices

### Performance
- Use `parallel=True` for CPU-intensive sync functions
- Pre-pull Docker image: `docker pull cheolwanpark/claude-agent-toolkit:0.2.2`
- Manage your own data explicitly - no hidden state

### Security
- Never hardcode OAuth tokens - use environment variables
- Validate all input parameters in tool methods
- Tools run on localhost by default

### Production
```bash
export CLAUDE_CODE_OAUTH_TOKEN='prod-token'
pip install claude-agent-toolkit

# Monitor with logging
set_logging(LogLevel.INFO, show_time=True)
```

## Framework Philosophy

This framework enables production Claude Code agents by:
1. Leveraging Claude Code's reasoning with your subscription token
2. Providing custom tools that extend Claude Code's capabilities
3. Explicit data management where users control persistence
4. Intelligent multi-tool orchestration