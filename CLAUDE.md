# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude-ADK is a production-ready Python framework for building sophisticated Claude Code agents with custom MCP tools. This framework provides a Docker-isolated environment where Claude Code can orchestrate stateful, parallel-processing tools for enterprise workflows, leveraging your Claude Code subscription token.

### Key Features
- **Production-Ready Architecture**: Enterprise-grade agent framework with proper state management
- **MCP Tool Integration**: HTTP-based Model Context Protocol servers with automatic discovery
- **Stateful Operations**: Advanced state management with versioning and conflict resolution
- **Parallel Processing**: CPU-bound operations run in separate worker processes
- **Docker Isolation**: Pre-built Docker environment ensures consistent execution
- **Intelligent Orchestration**: Claude Code makes context-aware decisions about tool usage

## Architecture

### System Overview
Claude-ADK implements a distributed architecture where custom MCP tools run as HTTP servers on the host machine, while Claude Code executes in an isolated Docker container that can communicate with these tools.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Host Machine  │    │ Docker Container │    │   Tool Processes    │
│                 │    │                  │    │                     │
│  Agent.run()    │───▶│  Claude Code CLI │───▶│ HTTP MCP Servers    │
│  ├─ToolConnector│    │  ├─ MCP Client   │    │ ├─ BaseTool         │
│  ├─DockerManager│    │  └─ Entrypoint   │    │ ├─ StateManager     │
│  └─ContainerExec│    │                  │    │ └─ WorkerManager    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

### Core Components

#### Agent Framework (`src/claude_adk/agent/`)
- **Agent** (`core.py`): Main orchestrator class managing the entire agent lifecycle
- **DockerManager** (`docker_manager.py`): Manages Docker images and container lifecycle
- **ContainerExecutor** (`executor.py`): Executes Claude Code commands in isolated containers
- **ToolConnector** (`tool_connector.py`): Manages connections and URLs for MCP tool servers

#### MCP Tool Framework (`src/claude_adk/tool/`)
- **BaseTool** (`base.py`): Abstract base class for all custom tools with state management
- **@tool decorator** (`decorator.py`): Method decorator that registers MCP tools with rich configuration
- **MCPServer** (`server.py`): HTTP server implementation using FastMCP with automatic port selection
- **StateManager** (`state_manager.py`): JSON-based state versioning with conflict resolution
- **WorkerManager** (`worker.py`): Process pool for CPU-bound operations with state snapshots

#### Docker Environment
- **Pre-built Image**: `cheolwanpark/claude-adk:latest` - Production Docker image with Claude Code CLI
- **Host Networking**: Container uses host network to access MCP tool servers
- **Entrypoint** (`src/docker/entrypoint.py`): Container initialization and MCP client configuration

### Agent Development Pattern

#### 1. Tool Development Lifecycle
```python
class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {"operations": [], "config": {}}
    
    @tool(
        description="Process data with advanced options",
        cpu_bound=True,  # Run in separate process
        timeout_s=120,   # Custom timeout
        conflict_policy="retry",  # State conflict handling
        max_retries=5    # Retry attempts
    )
    async def process_data(self, data: str, options: dict) -> dict:
        # Tool implementation
        return {"result": processed_data}
```

#### 2. State Management System
- **Automatic Versioning**: Every state change increments version counter
- **Conflict Detection**: JSON patch-based change tracking prevents race conditions
- **Retry Policies**: Configurable exponential backoff for state conflicts
- **Snapshot Support**: CPU-bound operations capture state snapshots for consistency

#### 3. MCP Server Lifecycle
1. Tool instantiation automatically creates MCPServer instance
2. Server selects available port and starts HTTP endpoint
3. Health checking ensures server readiness before agent execution
4. Graceful shutdown with proper resource cleanup

#### 4. Docker Orchestration Flow
1. Agent ensures Docker image availability (pulls if needed)
2. ToolConnector gathers all connected tool URLs
3. ContainerExecutor creates isolated container with tool access
4. Claude Code runs with full MCP tool integration
5. Results parsed and returned with execution metadata

## Development Commands

### Setup and Running
```bash
# Required: Set OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Install dependencies
uv sync

# Run the demonstration examples
# Calculator example:
cd src/examples/calculator && python main.py
# Weather example:
cd src/examples/weather && python main.py
```

### Package Management
```bash
# Add dependencies
uv add package_name

# Update lock file
uv lock --upgrade
```

## Key Dependencies

- Python 3.12+ with `uv` package manager
- Required packages: `docker`, `fastmcp`, `httpx`, `jsonpatch`, `uvicorn`
- Docker Desktop (must be running)
- Claude Code OAuth token from [Claude Code](https://claude.ai/code)
- Node.js 20 (installed in Docker environment)

## Development Workflow

1. **Start Docker Desktop** - Required for agent execution
2. **Set OAuth Token** - Export CLAUDE_CODE_OAUTH_TOKEN environment variable  
3. **Create Custom Tools** - Inherit from BaseTool and implement @tool methods
4. **Build Your Agent** - Use examples in `src/examples/` to see demonstrations or create custom agent scripts
5. **Deploy to Production** - Use your Claude Code subscription to run agents at scale

## Advanced Code Patterns

### Tool Creation with State Management
```python
from claude_adk import BaseTool, tool
from typing import List, Dict, Any
import time

class AdvancedTool(BaseTool):
    def __init__(self):
        super().__init__()
        # State is automatically managed with versioning
        self.state = {
            "operations": [],
            "cache": {},
            "config": {"max_operations": 100}
        }
    
    @tool(
        description="Fast async operation with state tracking",
        cpu_bound=False
    )
    async def quick_operation(self, data: str) -> Dict[str, Any]:
        # State changes are automatically versioned
        operation = {
            "timestamp": time.time(),
            "input": data,
            "result": f"processed_{data}"
        }
        self.state["operations"].append(operation)
        return operation
    
    @tool(
        description="CPU-intensive operation with worker process",
        cpu_bound=True,
        timeout_s=300,
        snapshot=["config"],  # Snapshot specific state fields
        conflict_policy="retry",
        max_retries=3
    )
    async def heavy_computation(self, numbers: List[int]) -> Dict[str, Any]:
        # This runs in a separate process with state snapshot
        import math
        results = [math.factorial(n) for n in numbers if n < 20]
        
        # State changes are merged back after process completion
        self.state["cache"]["last_computation"] = {
            "input_count": len(numbers),
            "results_count": len(results)
        }
        return {"factorials": results}
    
    @tool(
        description="Operation with advanced error handling",
        cpu_bound=False,
        conflict_policy="error",  # Fail fast on conflicts
    )
    async def critical_operation(self, value: str) -> Dict[str, Any]:
        try:
            # Simulate critical operation
            if not value:
                raise ValueError("Value cannot be empty")
            
            result = {"processed": value.upper(), "timestamp": time.time()}
            self.state["operations"].append(result)
            return result
            
        except Exception as e:
            # Error is automatically logged and propagated
            return {"error": str(e), "success": False}
```

### Agent Configuration and Usage
```python
from claude_adk import Agent
import asyncio

async def main():
    # Create agent with custom configuration
    agent = Agent(
        oauth_token="your-token",  # Or use CLAUDE_CODE_OAUTH_TOKEN env var
        system_prompt="You are a data processing assistant specialized in mathematical operations.",
        tools=[AdvancedTool()]  # Auto-connect tools during initialization
    )
    
    # Alternative: Connect tools after initialization
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    agent.connect(weather_tool)
    agent.connect(calculator_tool)
    
    # Execute with detailed response
    result = await agent.run(
        prompt="Calculate factorial of 5 and get weather for San Francisco",
        verbose=True  # Detailed execution logs
    )
    
    print(f"Success: {result['success']}")
    print(f"Response: {result['response']}")
    print(f"Execution time: {result['execution_time']}s")
    print(f"Tools used: {result['tools_used']}")

# Run the agent
asyncio.run(main())
```

### State Conflict Resolution Patterns
```python
class StatefulTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {"counter": 0, "items": []}
    
    @tool(
        description="Increment counter with retry on conflict",
        conflict_policy="retry",
        max_retries=5,
        backoff_initial_ms=10,
        backoff_max_ms=1000
    )
    async def increment_counter(self) -> Dict[str, Any]:
        # If state conflict occurs, this will automatically retry
        # with exponential backoff
        old_value = self.state["counter"]
        self.state["counter"] += 1
        return {"old_value": old_value, "new_value": self.state["counter"]}
    
    @tool(
        description="Add item with error on conflict",
        conflict_policy="error"  # Fail immediately on conflict
    )
    async def add_item(self, item: str) -> Dict[str, Any]:
        # This will raise an exception if state conflict occurs
        self.state["items"].append({
            "value": item,
            "timestamp": time.time()
        })
        return {"item_count": len(self.state["items"])}
```

### Error Handling and Debugging
```python
class RobustTool(BaseTool):
    @tool(description="Tool with comprehensive error handling")
    async def robust_method(self, data: str) -> Dict[str, Any]:
        try:
            # Your tool logic here
            result = self.process_data(data)
            return {"success": True, "result": result}
            
        except ValueError as e:
            # Handle specific exceptions
            return {"success": False, "error": "validation", "message": str(e)}
        except Exception as e:
            # Handle unexpected exceptions
            return {"success": False, "error": "unexpected", "message": str(e)}
    
    def process_data(self, data: str) -> str:
        if not data.strip():
            raise ValueError("Data cannot be empty or whitespace")
        return data.upper()
```

## Claude Code Agent Approach

This framework enables building production Claude Code agents by:
1. Leveraging Claude Code's advanced reasoning with your subscription token
2. Providing custom tools that extend Claude Code's capabilities
3. Managing stateful workflows where Claude Code builds context across tool interactions
4. Orchestrating multi-tool coordination through Claude Code's intelligent decision-making

The framework focuses on production-ready agent development rather than testing - you build agents that use Claude Code's intelligence with your custom tool implementations.

## API Reference

### Agent Class (`claude_adk.agent.Agent`)

```python
class Agent:
    def __init__(
        self,
        oauth_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None
    )
```

**Parameters:**
- `oauth_token`: Claude Code OAuth token (or use `CLAUDE_CODE_OAUTH_TOKEN` env var)
- `system_prompt`: Custom system prompt to modify agent behavior
- `tools`: List of tool instances to connect automatically

**Methods:**
- `connect(tool: BaseTool)`: Connect a tool instance to the agent
- `async run(prompt: str, verbose: bool = False) -> Dict[str, Any]`: Execute agent with given prompt

**Return Format:**
```python
{
    "success": bool,
    "response": str,           # Claude's response
    "execution_time": float,   # Seconds
    "tools_used": List[str],   # Tool names that were called
    "container_id": str,       # Docker container identifier
    "tool_calls": List[Dict]   # Detailed tool call logs
}
```

### BaseTool Class (`claude_adk.tool.BaseTool`)

```python
class BaseTool:
    def __init__(self):
        self.state: Dict[str, Any]  # Persistent state dictionary
```

**Properties:**
- `state`: Automatically managed state dictionary with versioning
- `connection_url`: HTTP URL for MCP server connection
- `health_url`: Health check endpoint URL

**Methods:**
- `run()`: Start the MCP server (called automatically)
- `cleanup()`: Stop server and cleanup resources

### @tool Decorator (`claude_adk.tool.tool`)

```python
def tool(
    name: Optional[str] = None,
    description: str = "",
    *,
    cpu_bound: bool = False,
    timeout_s: int = 60,
    snapshot: Optional[List[str]] = None,
    conflict_policy: str = "retry",
    max_retries: int = 16,
    backoff_initial_ms: int = 5,
    backoff_max_ms: int = 250,
)
```

**Parameters:**
- `name`: Tool name (defaults to function name)
- `description`: Tool description for Claude Code
- `cpu_bound`: Whether to run in separate process
- `timeout_s`: Timeout for CPU-bound operations
- `snapshot`: State fields to snapshot for CPU-bound ops
- `conflict_policy`: "retry" or "error" for state conflicts
- `max_retries`: Maximum retry attempts
- `backoff_initial_ms`: Initial backoff delay
- `backoff_max_ms`: Maximum backoff delay

### StateManager Class (`claude_adk.tool.StateManager`)

```python
class StateManager:
    def __init__(self, initial_state: Dict[str, Any])
    
    @property
    def state -> Dict[str, Any]  # Current state
    @property
    def version -> int           # Current version number
    
    def get_snapshot() -> Dict[str, Any]
    def apply_changes(changes: List[Dict]) -> None
    def clone_state() -> Dict[str, Any]
```

## Troubleshooting Guide

### Common Issues

#### 1. Docker Connection Errors
```bash
# Error: Cannot connect to Docker daemon
sudo systemctl start docker  # Linux
# Or start Docker Desktop on macOS/Windows
```

#### 2. OAuth Token Issues
```python
# Error: OAuth token required
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
# Or pass directly to Agent constructor
agent = Agent(oauth_token='your-token-here')
```

#### 3. Port Conflicts
```python
# Error: Port already in use
# Tools automatically select available ports, but you can specify:
class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self._port = 9000  # Force specific port
```

#### 4. State Conflicts
```python
# Handle state conflicts gracefully
@tool(conflict_policy="retry", max_retries=5)
async def my_method(self):
    # Will automatically retry on state conflicts
    pass
```

#### 5. CPU-Bound Operation Timeouts
```python
@tool(cpu_bound=True, timeout_s=300)  # 5 minute timeout
async def long_running_task(self):
    # Increase timeout for long operations
    pass
```

### Debug Mode
```python
# Enable verbose logging
result = await agent.run("your prompt", verbose=True)

# Check tool server health
tool = MyTool()
tool.run()  # Start server
# Visit http://localhost:{port}/health in browser
```

## Performance Optimization

### 1. CPU-Bound Operations
- Always use `cpu_bound=True` for computationally intensive tasks
- Use `snapshot` parameter to minimize state transfer overhead
- Set appropriate timeouts to prevent hanging

### 2. State Management
- Keep state dictionaries lightweight
- Use conflict resolution strategies appropriate for your use case
- Consider using `conflict_policy="error"` for critical operations

### 3. Docker Optimization
- Pre-pull the Docker image: `docker pull cheolwanpark/claude-adk:latest`
- Use Docker's host networking mode (done automatically)
- Monitor container resource usage in production

### 4. Tool Design
- Design tools to be stateless when possible
- Use async/await properly for I/O-bound operations
- Implement proper error handling and graceful degradation

## Security Considerations

### 1. OAuth Token Security
- Never hardcode OAuth tokens in source code
- Use environment variables or secure credential management
- Rotate tokens regularly according to Claude Code policies

### 2. Docker Security
- Review the pre-built Docker image security regularly
- Consider building custom images for production environments
- Implement proper container isolation in production

### 3. Tool Security
- Validate all input parameters in tool methods
- Implement rate limiting for resource-intensive operations
- Use proper error handling to avoid information leakage

### 4. State Security
- Avoid storing sensitive data in tool state
- Implement proper data sanitization
- Consider encryption for sensitive state data

### 5. Network Security
- Tools run HTTP servers on localhost by default
- Implement authentication if exposing tools externally
- Use HTTPS in production environments

## Production Deployment

### Environment Setup
```bash
# Production environment variables
export CLAUDE_CODE_OAUTH_TOKEN='prod-token'
export DOCKER_HOST='unix:///var/run/docker.sock'

# Install in production
pip install claude-adk

# Or using uv for faster installs
uv add claude-adk
```

### Monitoring and Logging
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Monitor tool performance
class MonitoredTool(BaseTool):
    @tool(description="Monitored operation")
    async def operation(self, data: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            result = await self.process_data(data)
            execution_time = time.time() - start_time
            logging.info(f"Operation completed in {execution_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"Operation failed: {e}")
            raise
```

### Scaling Considerations
- Use horizontal scaling with multiple agent instances
- Implement proper resource limits for Docker containers
- Monitor memory usage and implement cleanup strategies
- Consider using container orchestration platforms for large deployments