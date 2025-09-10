# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Agent Toolkit is a production-ready Python framework for building sophisticated Claude Code agents with custom MCP tools. This framework provides a Docker-isolated environment where Claude Code can orchestrate parallel-processing tools for enterprise workflows, leveraging your Claude Code subscription token.

### Key Features
- **Production-Ready Architecture**: Enterprise-grade agent framework with explicit data management
- **MCP Tool Integration**: HTTP-based Model Context Protocol servers with automatic discovery
- **Parallel Processing**: CPU-intensive operations run in separate worker processes
- **Simplified Architecture**: No hidden state management - users control their own data
- **Docker Isolation**: Pre-built Docker environment ensures consistent execution
- **Intelligent Orchestration**: Claude Code makes context-aware decisions about tool usage

## Architecture

### System Overview
Claude Agent Toolkit implements a flexible architecture supporting both Docker-isolated and direct subprocess execution. Custom MCP tools run as HTTP servers on the host machine, while Claude Code can execute either in an isolated Docker container or directly as a subprocess.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Host Machine                               │
│                                                                         │
│  Agent.run()                                                            │
│  ├─ ToolConnector                                                       │
│  └─ Executor (BaseExecutor)                                             │
│      │                                                                  │
│      ├─ DockerExecutor ───────┐                                         │
│      │                       │                                         │
│      └─ SubprocessExecutor ───┼───────┐                                 │
│                               │       │                                 │
│  Built-in Tools (tools/)      │       │                                 │
│  ├─ FileSystemTool            │       │                                 │
│  └─ [Future Tools]            │       │                                 │
│                               │       │                                 │
└───────────────────────────────┼───────┼─────────────────────────────────┘
                                │       │                                  
┌──────────────────────────────┼───────┼─────────────────────────────────┐
│       Docker Container        │       │    Tool Processes               │
│                              │       │                                 │
│  Claude Code CLI ◀───────────┘       │    HTTP MCP Servers             │
│  ├─ MCP Client                       │    ├─ BaseTool                  │
│  └─ Entrypoint                       │    └─ WorkerManager             │
│                                      │                                 │
└──────────────────────────────────────┼─────────────────────────────────┘
                                       │                                  
┌──────────────────────────────────────┼─────────────────────────────────┐
│      Direct Subprocess               │                                 │
│                                      │                                 │
│  claude-code-sdk ◀───────────────────┘                                 │
│  ├─ MCP Client                                                          │
│  └─ ResponseHandler                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

#### Agent Framework (`src/claude_agent_toolkit/agent/`)
- **Agent** (`core.py`): Main orchestrator class managing the entire agent lifecycle
- **BaseExecutor** (`executor/base.py`): Abstract base class defining the executor interface
- **DockerExecutor** (`executor/docker.py`): Docker-based execution with integrated container management
- **SubprocessExecutor** (`executor/subprocess.py`): Direct subprocess execution using claude-code-sdk
- **ResponseHandler** (`response_handler.py`): Handles streaming responses from executors
- **ToolConnector** (`tool_connector.py`): Manages connections and URLs for MCP tool servers

#### MCP Tool Framework (`src/claude_agent_toolkit/tool/`)
- **BaseTool** (`base.py`): Abstract base class for all custom tools
- **@tool decorator** (`decorator.py`): Method decorator that registers MCP tools with rich configuration
- **MCPServer** (`server.py`): HTTP server implementation using FastMCP with automatic port selection
- **WorkerManager** (`worker.py`): ProcessPoolExecutor manager for parallel operations

#### Built-in Tools (`src/claude_agent_toolkit/tools/`)
- **Purpose**: Pre-built tool implementations ready for immediate use
- **Structure**: Each tool is a standalone module inheriting from BaseTool
- **Import Pattern**: Clean imports via package `__init__.py` exports
- **Development**: Follow same patterns as custom tools in examples

#### Executor Architecture

The framework supports two execution modes through a flexible executor abstraction:

**DockerExecutor** (Default):
- **Pre-built Image**: `cheolwanpark/claude-agent-toolkit:0.1.4` - Production Docker image with Claude Code CLI
- **Host Networking**: Container uses host network to access MCP tool servers
- **Integrated Management**: Combines Docker image management with container execution
- **Entrypoint** (`src/docker/entrypoint.py`): Container initialization and MCP client configuration
- **Use Cases**: Production environments, full isolation, consistent execution environment

**SubprocessExecutor** (Alternative):
- **Direct Execution**: Uses `claude-code-sdk` directly without Docker dependency
- **Faster Startup**: ~6x faster startup time (0.5s vs 3s)
- **Temporary Isolation**: Creates temporary directories for minimal file system isolation
- **Async Context Management**: Isolated async contexts prevent TaskGroup violations
- **Use Cases**: Development, testing, environments without Docker, lightweight execution

#### Docker Environment (DockerExecutor)
- **Production Image**: `cheolwanpark/claude-agent-toolkit:0.1.4`
- **Version Safety**: Automatic version matching between package and Docker image
- **Network Configuration**: Host networking with localhost URL rewriting
- **Entrypoint**: Container initialization with MCP client setup

### Agent Development Pattern

#### 1. Tool Development Lifecycle
```python
class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        # Explicit data management - no automatic state
        self.operations = []
        self.config = {}
    
    @tool(
        description="Process data asynchronously",
        # parallel=False by default for async functions
    )
    async def process_data_async(self, data: str, options: dict) -> dict:
        # Async tool implementation
        return {"result": processed_data}
    
    @tool(
        description="Process data in parallel",
        parallel=True,   # Run in separate process
        timeout_s=120    # Custom timeout for parallel ops
    )
    def process_data_parallel(self, data: str, options: dict) -> dict:
        # Sync tool implementation - runs in worker process
        return {"result": processed_data}
```

#### 2. Data Management
- **Explicit Control**: Users manage their own data as instance variables
- **No Hidden State**: No automatic versioning or conflict resolution
- **ProcessPoolExecutor Isolation**: Parallel operations (parallel=True) run under ProcessPoolExecutor with new instances
- **Critical**: Users MUST use semaphores or atomic datatypes for parallel=True tools when sharing data

#### 3. MCP Server Lifecycle
1. Tool instantiation automatically creates MCPServer instance
2. Server selects available port and starts HTTP endpoint
3. Health checking ensures server readiness before agent execution
4. Graceful shutdown with proper resource cleanup

#### 4. Executor Selection and Flow

**Executor Selection**:
```python
from claude_agent_toolkit import Agent, ExecutorType

# Use Docker executor (default)
agent = Agent(tools=[my_tool])  # or executor=ExecutorType.DOCKER

# Use Subprocess executor 
agent = Agent(tools=[my_tool], executor=ExecutorType.SUBPROCESS)
```

**Docker Execution Flow**:
1. DockerExecutor ensures Docker image availability (pulls if needed)
2. ToolConnector gathers connected tool URLs with Docker host mapping
3. Docker container created with isolated environment and tool access
4. Claude Code CLI runs in container with full MCP tool integration
5. ResponseHandler processes streaming results and returns response

**Subprocess Execution Flow**:
1. SubprocessExecutor creates temporary directory for isolation
2. ToolConnector gathers tool URLs without Docker host mapping
3. claude-code-sdk executed directly in isolated async context
4. MCP tools accessed via localhost URLs
5. ResponseHandler processes streaming results and returns response

### Built-in Tools Development Pattern

The framework includes pre-built tools in `src/claude_agent_toolkit/tools/` that follow the same development patterns as custom tools:

#### Directory Structure
```
src/claude_agent_toolkit/tools/
├── __init__.py          # Package exports for clean imports
├── filesystem.py        # FileSystemTool implementation
└── [future_tools.py]    # Additional built-in tools
```

#### Import Conventions
```python
# Built-in tools use clean package imports
from claude_agent_toolkit.tools import FileSystemTool

# Custom tools use direct module imports
from my_project.tools import MyCustomTool
```

#### Development Guidelines
- **Same Patterns**: Built-in tools follow identical patterns to examples
- **Standalone Modules**: Each tool is self-contained with minimal dependencies
- **BaseTool Inheritance**: All tools inherit from the same BaseTool class
- **Standard Decorators**: Use the same @tool decorator with consistent parameters

## Development Commands

### Setup and Running
```bash
# Required: Set OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Install dependencies
uv sync

# Run the demonstration examples
# Calculator example (Docker executor):
cd src/examples/calculator && python main.py
# Weather example (Docker executor):
cd src/examples/weather && python main.py
# Subprocess example (no Docker required):
cd src/examples/subprocess && python main.py
# Filesystem example (demonstrates built-in tools):
cd src/examples/filesystem && python main.py
```

## Configuration

### Constants File (`src/claude_agent_toolkit/constants.py`)

The framework uses a centralized constants file for shared configuration values used across multiple modules:

```python
# Docker networking configuration
DOCKER_LOCALHOST_MAPPINGS = {
    "localhost": "host.docker.internal",
    "127.0.0.1": "host.docker.internal",
}
DOCKER_HOST_GATEWAY = "host-gateway"

# Environment variable names
ENV_CLAUDE_CODE_OAUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"

# Container naming
CONTAINER_NAME_PREFIX = "agent-"
CONTAINER_UUID_LENGTH = 8

# Model ID mappings (short aliases to full model IDs)
MODEL_ID_MAPPING = {
    "opus": "claude-opus-4-1-20250805",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022"
}
```

**Constants Policy**: Only values used across multiple modules are centralized. Function defaults, version numbers, and single-use values remain in their original locations following Python best practices.

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

## Release Process

### Testing Strategy
Use pre-release tags to test Docker builds before PyPI release:
```bash
# Test Docker build with beta tag (PEP 440 compliant)
git tag v0.1.2b1
git push origin v0.1.2b1
# Workflow builds Docker image only (PyPI fails safely)

# After testing, create official release
git tag v0.1.2
git push origin v0.1.2
# Both Docker and PyPI succeed
```

### PyPI Trusted Publishing
This project uses GitHub OIDC for secure PyPI publishing:
- No API tokens stored in GitHub Secrets
- Automatic authentication via GitHub Actions
- Configure at: pypi.org → package → Publishing → Add GitHub publisher

#### PyPI Configuration Steps
1. Go to [PyPI](https://pypi.org) and log in
2. Navigate to **claude-agent-toolkit** package settings
3. Go to **Publishing** section
4. Add GitHub Publisher with these exact values:
   - **Owner**: cheolwanpark
   - **Repository**: claude-agent-toolkit
   - **Workflow name**: release.yml
   - **Environment name**: (leave blank)

### Release Workflow
The unified `release.yml` workflow handles:
1. Python package build and PyPI publishing (via trusted publisher)
2. Docker multi-platform image build and push
3. Automatic version extraction and tagging

### Release Commands
```bash
# Pre-release testing (PEP 440: a1=alpha, b1=beta, rc1=release candidate)
git tag v0.1.2b1 && git push origin v0.1.2b1

# Official release
git tag v0.1.2 && git push origin v0.1.2
```

## Advanced Code Patterns

### Tool Creation with Explicit Data Management
```python
from claude_agent_toolkit import BaseTool, tool
from typing import List, Dict, Any
import time

class AdvancedTool(BaseTool):
    def __init__(self):
        super().__init__()
        # Explicit data management - users control their own data
        self.operations = []
        self.cache = {}
        self.config = {"max_operations": 100}
    
    @tool(description="Fast async operation")
    async def quick_operation(self, data: str) -> Dict[str, Any]:
        # Manage your own data explicitly
        operation = {
            "timestamp": time.time(),
            "input": data,
            "result": f"processed_{data}"
        }
        self.operations.append(operation)
        return operation
    
    @tool(
        description="CPU-intensive operation with worker process",
        parallel=True,   # Runs in separate process
        timeout_s=300    # Custom timeout for parallel operations
    )
    def heavy_computation(self, numbers: List[int]) -> Dict[str, Any]:
        # Sync function - runs in separate process
        # Note: ProcessPoolExecutor creates new instance, self.cache won't persist
        # Users must use semaphores or atomic datatypes if sharing data
        import math
        results = [math.factorial(n) for n in numbers if n < 20]
        
        return {
            "factorials": results,
            "input_count": len(numbers),
            "results_count": len(results)
        }
    
    @tool(description="Operation with error handling")
    async def critical_operation(self, value: str) -> Dict[str, Any]:
        try:
            # Async operation
            if not value:
                raise ValueError("Value cannot be empty")
            
            result = {"processed": value.upper(), "timestamp": time.time()}
            self.operations.append(result)
            return result
            
        except Exception as e:
            # Error is automatically logged and propagated
            return {"error": str(e), "success": False}
```

### Agent Configuration and Usage
```python
from claude_agent_toolkit import Agent, ExecutorType
import asyncio

async def main():
    # Create agent with Docker executor (default)
    agent = Agent(
        oauth_token="your-token",  # Or use CLAUDE_CODE_OAUTH_TOKEN env var
        system_prompt="You are a data processing assistant specialized in mathematical operations.",
        tools=[AdvancedTool()],  # Auto-connect tools during initialization
        model="haiku",  # Use fast Haiku model for simple tasks
        executor=ExecutorType.DOCKER  # Optional - Docker is default
    )
    
    # Alternative: Create agent with Subprocess executor
    subprocess_agent = Agent(
        oauth_token="your-token",
        system_prompt="You are a development assistant.",
        tools=[AdvancedTool()],
        model="haiku",
        executor=ExecutorType.SUBPROCESS  # No Docker required
    )
    
    # Alternative: Connect tools after initialization
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    agent.connect(weather_tool)
    agent.connect(calculator_tool)
    
    # Execute with detailed response
    result = await agent.run(
        prompt="Calculate factorial of 5 and get weather for San Francisco",
        verbose=True,  # Detailed execution logs
        model="opus"  # Override to use Opus model for complex calculation
    )
    
    print(f"Response: {result}")  # Agent.run() returns string directly

# Run the agent
asyncio.run(main())
```

### Async vs Parallel Tool Patterns
```python
class StatefulTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {"counter": 0, "items": []}
    
    @tool(description="Increment counter")
    async def increment_counter(self) -> Dict[str, Any]:
        # Users manage their own data explicitly
        old_value = self.state["counter"]
        self.state["counter"] += 1
        return {"old_value": old_value, "new_value": self.state["counter"]}
    
    @tool(description="Add item to list")
    async def add_item(self, item: str) -> Dict[str, Any]:
        # Users handle concurrency explicitly if needed
        self.state["items"].append({
            "value": item,
            "timestamp": time.time()
        })
        return {"item_count": len(self.state["items"])}
```

## Subprocess Executor

Claude Agent Toolkit provides a SubprocessExecutor as an alternative to Docker execution, offering faster startup and eliminating Docker dependency.

### Benefits and Use Cases

**Performance Benefits:**
- **6x Faster Startup**: ~0.5 seconds vs ~3 seconds for Docker
- **Lower Resource Usage**: No container overhead
- **Direct SDK Integration**: Uses claude-code-sdk directly on the host

**Development Benefits:**
- **No Docker Dependency**: Works without Docker Desktop
- **Easier Debugging**: Direct process execution for simpler troubleshooting
- **Faster Iteration**: Quick testing and development cycles

**When to Use SubprocessExecutor:**
- Development and testing environments
- CI/CD pipelines without Docker
- Lightweight execution scenarios
- Environments where Docker is not available
- Quick prototyping and experimentation

**When to Use DockerExecutor:**
- Production deployments
- Maximum isolation requirements
- Consistent execution environments
- Security-sensitive applications

### Usage Example

```python
from claude_agent_toolkit import Agent, ExecutorType

# Subprocess executor - no Docker required
agent = Agent(
    tools=[my_tool],
    executor=ExecutorType.SUBPROCESS  # Key difference
)

result = await agent.run("Your prompt here")
```

### Comparison Table

| Feature | Docker Executor | Subprocess Executor |
|---------|----------------|--------------------|
| **Dependencies** | Docker Desktop | claude-code-sdk only |
| **Startup Time** | ~3 seconds | ~0.5 seconds |
| **Resource Usage** | Higher (container) | Lower (direct process) |
| **Isolation** | Full container isolation | Process + temp directory |
| **Use Case** | Production, security | Development, testing |
| **Debugging** | Container logs | Direct process output |

## Model Selection

Claude Agent Toolkit supports flexible model selection, allowing you to choose the best Claude model for your specific use case:

### Available Models
- **"haiku"** - Fast and efficient for simple tasks, low cost
- **"sonnet"** - Balanced performance and capability  
- **"opus"** - Most capable model for complex reasoning

### Usage Examples

#### Set Default Model
```python
from claude_agent_toolkit import Agent

# Use Haiku for simple, fast operations
weather_agent = Agent(
    system_prompt="You are a weather assistant",
    tools=[weather_tool],
    model="haiku"
)

# Use Opus for complex analysis
analysis_agent = Agent(
    system_prompt="You are a data analyst",
    tools=[analysis_tool], 
    model="opus"
)
```

#### Override Model Per Run
```python
# Start with default model
agent = Agent(model="sonnet")

# Use different models for specific tasks
simple_result = await agent.run("What's 2+2?", model="haiku")
complex_result = await agent.run("Analyze this dataset", model="opus")
```

#### Full Model IDs
```python
# You can also use specific model IDs
agent = Agent(model="claude-3-5-haiku-20241022")
agent = Agent(model="claude-opus-4-1-20250805")
```

### Model Selection Guidelines
- **Haiku**: Simple queries, basic calculations, fast responses needed
- **Sonnet**: General purpose, balanced tasks, good default choice
- **Opus**: Complex reasoning, detailed analysis, maximum capability needed

## Docker Image Version Safety

Claude Agent Toolkit enforces strict version matching between PyPI package and Docker image for maximum safety and compatibility.

### Automatic Version Matching
- **Safety First**: Docker image version **automatically matches** the installed package version (`__version__`)
- **No Configuration**: No parameters needed - version matching is enforced internally
- **No Fallback**: Exact version match required - no fallback for maximum safety

### Version Consistency Examples

```python
from claude_agent_toolkit import Agent

# All these examples use the SAME Docker image version
# matching the installed claude-agent-toolkit version

agent = Agent(oauth_token="your-token")
# ✅ Uses: cheolwanpark/claude-agent-toolkit:0.1.4 (if package version is 0.1.4)

agent = Agent(
    system_prompt="You are a helpful assistant",
    tools=[my_tool]
)
# ✅ Uses: cheolwanpark/claude-agent-toolkit:0.1.4 (same version as above)
```

### Safety Benefits
- **Perfect Version Consistency**: PyPI and Docker versions always match exactly
- **No Version Conflicts**: Eliminates all compatibility issues
- **Predictable Behavior**: Same package version = identical Docker environment
- **Production Safe**: No accidental version mismatches or unexpected fallbacks

### Strict Version Enforcement
- **Exact Match Required**: Only the specific version Docker image will be used
- **No Fallback Policy**: If the exact version is unavailable, the operation fails safely
- **Clear Error Messages**: Provides specific guidance when version-specific image is missing
- **Zero Tolerance**: No compromise on version consistency for maximum reliability

### Error Handling and Exception Management

Claude Agent Toolkit uses a comprehensive exception hierarchy for clear error handling:

```python
from claude_agent_toolkit import (
    BaseTool, tool, 
    ClaudeAgentError, ConfigurationError, ConnectionError, 
    ExecutionError
)

class RobustTool(BaseTool):
    @tool(description="Tool with comprehensive error handling")
    async def robust_method(self, data: str) -> Dict[str, Any]:
        try:
            # Your tool logic here
            result = self.process_data(data)
            return {"success": True, "result": result}
            
        except ValueError as e:
            # Convert domain exceptions to library exceptions
            raise ExecutionError(f"Data validation failed: {e}") from e
        except Exception as e:
            # Wrap unexpected exceptions
            raise ExecutionError(f"Tool execution failed: {e}") from e
    
    def process_data(self, data: str) -> str:
        if not data.strip():
            raise ValueError("Data cannot be empty or whitespace")
        return data.upper()

# Exception Handling Best Practices for Tool Developers:
class ExceptionAwareTool(BaseTool):
    def __init__(self):
        super().__init__()
        # Invalid configuration should raise ConfigurationError
        if not self.validate_config():
            raise ConfigurationError("Tool configuration is invalid")
    
    @tool(description="Method that can raise various exceptions")
    async def complex_method(self, param: str) -> Dict[str, Any]:
        # Let framework exceptions propagate naturally
        if not self._port:  # This will raise ConnectionError from BaseTool
            return self.connection_url
        
        # Wrap third-party exceptions appropriately
        try:
            import requests
            response = requests.get(f"https://api.example.com/{param}")
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to external API: {e}") from e
        except requests.exceptions.RequestException as e:
            raise ExecutionError(f"API request failed: {e}") from e
        
        return {"status": "success"}
    
    def validate_config(self) -> bool:
        # Configuration validation logic
        return True
```

### Exception Hierarchy Reference

```python
ClaudeAgentError                    # Base exception for all library errors
├── ConfigurationError             # Missing/invalid configuration
│   ├── Missing OAuth tokens       # Agent initialization failures  
│   └── Invalid tool configuration # Tool setup errors
├── ConnectionError                # Network and service connectivity
│   ├── Docker connection issues   # Docker daemon problems
│   ├── Socket binding failures    # Port conflicts, network issues
│   └── Tool server connectivity   # MCP server health failures
└── ExecutionError                 # Agent and tool execution failures
    ├── Tool method failures       # Custom tool errors
    ├── Parallel timeouts          # Worker process timeouts
    └── Agent execution issues     # Claude Code execution problems
```

## Claude Code Agent Approach

This framework enables building production Claude Code agents by:
1. Leveraging Claude Code's advanced reasoning with your subscription token
2. Providing custom tools that extend Claude Code's capabilities
3. Providing explicit data management where users control their own persistence
4. Orchestrating multi-tool coordination through Claude Code's intelligent decision-making

The framework focuses on production-ready agent development rather than testing - you build agents that use Claude Code's intelligence with your custom tool implementations.

## API Reference

### Agent Class (`claude_agent_toolkit.agent.Agent`)

```python
class Agent:
    def __init__(
        self,
        oauth_token: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None,
        executor: Optional[ExecutorType] = None
    )
```

**Parameters:**
- `oauth_token`: Claude Code OAuth token (or use `CLAUDE_CODE_OAUTH_TOKEN` env var)
- `system_prompt`: Custom system prompt to modify agent behavior
- `tools`: List of tool instances to connect automatically
- `model`: Model to use ("opus", "sonnet", "haiku", or any Claude model name/ID)
- `executor`: Executor type to use (ExecutorType.DOCKER or ExecutorType.SUBPROCESS, defaults to DOCKER)

**Note:** Docker image version automatically matches the installed package version for safety.

**Backward Compatibility:**
The framework maintains full backward compatibility:
- `ContainerExecutor` is aliased to `DockerExecutor` for existing code
- Docker executor remains the default behavior
- No breaking changes - existing code continues to work unchanged

**Methods:**
- `connect(tool: BaseTool)`: Connect a tool instance to the agent
- `async run(prompt: str, verbose: bool = False, model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None) -> str`: Execute agent with given prompt

**Returns:**
The `run()` method returns Claude's response as a string directly. For debugging and monitoring, use `verbose=True` to see detailed execution information printed to the console.

### BaseTool Class (`claude_agent_toolkit.tool.BaseTool`)

```python
class BaseTool:
    def __init__(self):
        # No automatic state - manage your own data as instance variables
```

**Properties:**
- `connection_url`: HTTP URL for MCP server connection
- `health_url`: Health check endpoint URL

**Methods:**
- `__del__()`: Automatically cleanup server resources when tool is destroyed

### @tool Decorator (`claude_agent_toolkit.tool.tool`)

```python
def tool(
    name: Optional[str] = None,
    description: str = "",
    *,
    parallel: bool = False,
    timeout_s: int = 60,
)
```

**Parameters:**
- `name`: Tool name (defaults to function name)
- `description`: Tool description for Claude Code
- `parallel`: Whether to run in separate process (must be sync function)
- `timeout_s`: Timeout for parallel operations

**Validation Rules:**
- `parallel=True` requires sync function (`def`, not `async def`)
- `parallel=False` requires async function (`async def`)



### Exception Classes (`claude_agent_toolkit.exceptions`)

Claude Agent Toolkit provides a comprehensive exception hierarchy for clear error handling:

```python
from claude_agent_toolkit import (
    ClaudeAgentError, ConfigurationError, ConnectionError,
    ExecutionError, ExecutorType
)
```

**Exception Hierarchy:**

```python
class ClaudeAgentError(Exception):
    """Base exception for all claude-agent-toolkit errors."""
    pass

class ConfigurationError(ClaudeAgentError):
    """Raised when configuration is missing or invalid."""
    pass

class ConnectionError(ClaudeAgentError):  
    """Raised when connection to services fails."""
    pass

class ExecutionError(ClaudeAgentError):
    """Raised when agent or tool execution fails."""
    pass

```

**When Each Exception is Raised:**

- **ConfigurationError**: Missing OAuth tokens, invalid tool configurations
- **ConnectionError**: Docker daemon issues, port conflicts, network failures, MCP server health failures
- **ExecutionError**: Tool method failures, agent execution problems, parallel operation timeouts

**Usage Examples:**
```python
# Catch specific exception types
try:
    agent = Agent(oauth_token="invalid")
    result = await agent.run("Calculate 2+2")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except ConnectionError as e:
    print(f"State management issue: {e}")

# Catch all library exceptions  
try:
    # Agent operations
    pass
except ClaudeAgentError as e:
    print(f"Library error: {e}")
```

### Logging Functions (`claude_agent_toolkit.logging`)

```python
from claude_agent_toolkit import set_logging, LogLevel

def set_logging(
    level: Union[LogLevel, str] = LogLevel.WARNING,
    format: Optional[str] = None,
    stream: TextIO = sys.stderr,
    show_time: bool = False,
    show_level: bool = False
) -> None
```

**Parameters:**
- `level`: Log level (LogLevel enum or string: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
- `format`: Custom format string (overrides show_* options)
- `stream`: Output stream (sys.stdout or sys.stderr)
- `show_time`: Add timestamp to log messages
- `show_level`: Add log level to log messages

**LogLevel Enum:**
```python
class LogLevel(str, Enum):
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'     # Default level
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
```

**Usage Examples:**
```python
# Enable INFO logging for development
set_logging(LogLevel.INFO)

# Debug mode with full details
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Production logging to stdout
set_logging(LogLevel.ERROR, stream=sys.stdout)

# Custom format
set_logging(format='%(asctime)s - %(name)s - %(message)s')
```

## Troubleshooting Guide

### Exception-Based Error Handling

Claude Agent Toolkit uses specific exception types to help you identify and handle errors:

```python
from claude_agent_toolkit import (
    Agent, BaseTool, tool,
    ClaudeAgentError, ConfigurationError, ConnectionError,
    ExecutionError
)

# Comprehensive error handling example
try:
    # Create agent and run
    agent = Agent(
        oauth_token="your-token",
        system_prompt="You are a helpful assistant",
        tools=[MyTool()]
    )
    result = await agent.run("Process my data")
    
except ConfigurationError as e:
    print(f"❌ Configuration Error: {e}")
    print("💡 Check OAuth token and tool configuration")
    
except ConnectionError as e:
    print(f"❌ Connection Error: {e}")
    if "Docker" in str(e):
        print("💡 Start Docker Desktop and try again")
    elif "bind" in str(e):
        print("💡 Port may be in use, wait and retry")
    else:
        print("💡 Check network connectivity")
        
except ExecutionError as e:
    print(f"❌ Execution Error: {e}")
    print("💡 Check tool implementation and agent logic")
    
except ConnectionError as e:
    print(f"❌ Connection Error: {e}")
    print("💡 Check tool lifecycle and server connection")
    
except ClaudeAgentError as e:
    print(f"❌ Library Error: {e}")
    print("💡 General library issue - check documentation")
    
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    print("💡 This may indicate a bug - please report")
```

### Common Issues and Solutions

#### 1. ConfigurationError - Missing Configuration
```python
# Error: OAuth token required
try:
    agent = Agent()  # No token provided
except ConfigurationError as e:
    print(f"Configuration issue: {e}")

# Solutions:
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
# Or pass directly:
agent = Agent(oauth_token='your-token-here')
```

#### 2. ConnectionError - Service Connectivity
```python
# Docker connection issues
try:
    agent = Agent()
except ConnectionError as e:
    if "Docker" in str(e):
        # Start Docker Desktop
        # Linux: sudo systemctl start docker
        pass
        
# Port binding issues  
try:
    tool = MyTool(port=8000)
except ConnectionError as e:
    if "bind" in str(e):
        # Port 8000 already in use - let tool auto-select
        tool = MyTool()  # Auto-selects available port
```

#### 3. ExecutionError - Runtime Failures  
```python
# Tool execution failures
try:
    result = await agent.run("Complex task")
except ExecutionError as e:
    if "timeout" in str(e):
        # Increase timeout for parallel operations
        @tool(parallel=True, timeout_s=300)
        def long_task(self):
            pass
    else:
        # Check tool implementation
        pass
```

#### 4. StateError - Data Management Issues
```python
# Tool lifecycle is now automatic
try:
    tool = MyTool()  # Server starts automatically
    url = tool.connection_url  # Always accessible after construction
except ConnectionError as e:
    print("Tool server failed to start:")
    print(f"Error: {e}")
    # Check Docker status and port availability

# State conflicts  
@tool(conflict_policy="retry", max_retries=5)
async def concurrent_method(self):
    # Automatically handles state conflicts with retries
    pass
```

#### 5. Debug Mode and Logging
```python
from claude_agent_toolkit import set_logging, LogLevel

# Enable detailed debug logging
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Try operation with full logging
try:
    result = await agent.run("your prompt", verbose=True)
except ClaudeAgentError as e:
    print(f"Detailed error with logging enabled: {e}")

# Configure logging for production
set_logging(LogLevel.ERROR, stream=sys.stdout)
```

### Debug Mode
```python
from claude_agent_toolkit import Agent, set_logging, LogLevel

# Enable debug logging to see detailed framework operations
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Enable verbose agent execution
result = await agent.run("your prompt", verbose=True)

# Check tool server health
tool = MyTool()  # Server starts automatically
# Visit http://localhost:{port}/health in browser

# Example debug output:
# 2025-01-15 10:30:45,123 INFO     [claude_agent_toolkit.agent] Running with prompt: Calculate...
# 2025-01-15 10:30:45,124 INFO     [claude_agent_toolkit.agent] Connected tools: ['CalculatorTool']
# 2025-01-15 10:30:45,125 DEBUG    [claude_agent_toolkit.agent] Starting container agent-abc12345
# 2025-01-15 10:30:45,200 INFO     [claude_agent_toolkit.tool] CalculatorTool @ http://127.0.0.1:50123/mcp
```

## Performance Optimization

### 1. Parallel Operations
- Use `parallel=True` for computationally intensive tasks (must be sync functions)
- Set appropriate timeouts to prevent hanging
- Handle your own data persistence - parallel operations create new instances

### 2. Data Management
- Manage your own data as instance variables
- Handle race conditions explicitly when needed  
- Consider data persistence requirements for parallel operations

### 3. Docker Optimization
- Pre-pull the Docker image: `docker pull cheolwanpark/claude-agent-toolkit:0.1.4`
- Use Docker's host networking mode (done automatically)
- Monitor container resource usage in production

### 4. Tool Design
- Tools are stateless by design - manage your own data explicitly
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

### 4. Data Security
- Avoid storing sensitive data in instance variables
- Implement proper data sanitization
- Consider encryption for sensitive data

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
pip install claude-agent-toolkit

# Or using uv for faster installs
uv add claude-agent-toolkit
```

### Logging Configuration

Claude Agent Toolkit includes a built-in logging system that follows Python library best practices. By default, the library stays quiet (WARNING level only), but you can configure it for development and production needs.

```python
from claude_agent_toolkit import Agent, set_logging, LogLevel

# Default: Library stays quiet (WARNING level, stderr)
agent = Agent()

# Enable INFO level logging for development
set_logging(LogLevel.INFO)

# Debug mode with timestamps
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Production logging to stdout
set_logging(LogLevel.WARNING, stream=sys.stdout)

# Custom format
set_logging(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

**Available Log Levels:**
- `LogLevel.DEBUG`: Detailed diagnostic information
- `LogLevel.INFO`: General operational messages
- `LogLevel.WARNING`: Warning messages (default)
- `LogLevel.ERROR`: Error messages
- `LogLevel.CRITICAL`: Critical system failures

**Logging Components:**
- `claude_agent_toolkit.agent`: Agent orchestration, Docker operations, tool connections
- `claude_agent_toolkit.tool`: Tool server startup, parallel execution

### Monitoring and Performance
```python
from claude_agent_toolkit import BaseTool, tool, set_logging, LogLevel
import time

# Enable detailed logging for monitoring
set_logging(LogLevel.INFO, show_time=True)

class MonitoredTool(BaseTool):
    @tool(description="Monitored operation with logging")
    async def operation(self, data: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            result = await self.process_data(data)
            execution_time = time.time() - start_time
            # Use tool's built-in logger (automatically available)
            return {"result": result, "execution_time": execution_time}
        except Exception as e:
            # Errors are automatically logged by the framework
            return {"success": False, "error": str(e)}
```

### Scaling Considerations
- Use horizontal scaling with multiple agent instances
- Implement proper resource limits for Docker containers
- Monitor memory usage and implement cleanup strategies
- Consider using container orchestration platforms for large deployments