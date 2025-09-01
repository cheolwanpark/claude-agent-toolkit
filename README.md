# Claude Code Agent Toolkit (claude-agent-toolkit)

A Python framework for building Claude Code agents with custom tools, designed to leverage Claude Code's advanced reasoning capabilities with your subscription token. The framework provides Docker-isolated environments where Claude Code can orchestrate custom MCP tools for production workflows.

## Key Features

- **Claude Code Integration** - Leverage Claude Code's advanced reasoning with your existing subscription token
- **Docker Isolation** - Complete isolation of agent execution environment with Claude Code CLI
- **Advanced State Management** - JSON patch-based state management with conflict resolution and automatic retries  
- **CPU-bound Operations** - Support for CPU-intensive operations with process pools and parallel execution
- **Multi-tool Coordination** - Claude Code orchestrates multiple tools in complex workflows
- **Production Ready** - Build scalable agents using Claude Code's capabilities with custom tool integration

## Architecture

### Core Components

- **Agent Framework** (`src/claude_agent_toolkit/agent/`) - Docker-isolated Agent class that runs Claude Code with MCP tool support
- **MCP Tool Framework** (`src/claude_agent_toolkit/tool/`) - BaseTool class for creating custom MCP tools with state management
- **Example Tools** (`src/examples/`) - Demonstration tools showing practical agent development patterns
- **Docker Environment** (`src/docker/`) - Isolated environment with Claude Code CLI and dependencies

## Quick Start

### Prerequisites

- **Python 3.12+** with `uv` package manager
- **Docker Desktop** (must be running)
- **Claude Code OAuth Token** - Get from [Claude Code](https://claude.ai/code)

### Installation

```bash
# Using pip
pip install claude-agent-toolkit

# Using uv
uv add claude-agent-toolkit

# Using poetry
poetry add claude-agent-toolkit

# Set your OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### Run the Demo

```bash
# Clone the repository for examples
git clone https://github.com/cheolwanpark/claude-agent-toolkit.git
cd claude-agent-toolkit

# Start Docker Desktop first, then run the verification demo
# Run calculator example:
cd src/examples/calculator && python main.py
# Run weather example:
cd src/examples/weather && python main.py
```

This will run demonstration examples:
1. **Calculator Demo** - Shows stateful mathematical operations and problem solving
2. **Weather Demo** - Demonstrates external API integration with real-time data

## Tool Development

### Creating Custom Tools

Create tools by inheriting from `BaseTool` and using the `@tool()` decorator:

```python
from claude_agent_toolkit import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {"counter": 0}
    
    @tool(description="Increment counter and return new value")
    async def increment(self) -> dict:
        self.state["counter"] += 1
        return {"value": self.state["counter"]}
    
    @tool(description="Heavy computation", cpu_bound=True)  
    def compute_heavy(self, data: str) -> dict:
        # CPU-intensive operation runs in process pool
        import time
        time.sleep(2)  # Simulate heavy computation
        return {"processed": f"Heavy result for {data}"}
```

### Using Tools with Agents

```python
from claude_agent_toolkit import Agent, ConnectionError, ExecutionError

try:
    # Create and start tool
    my_tool = MyTool().run(workers=2)

    # New pattern (recommended) - cleaner initialization
    agent = Agent(
        system_prompt="You are a helpful assistant specialized in calculations",
        tools=[my_tool]
    )

    # Traditional pattern - still supported
    # agent = Agent()
    # agent.connect(my_tool)

    # Run agent with prompt (verbose=True shows detailed message processing)
    result = await agent.run(
        "Please increment the counter twice and tell me the result",
        verbose=True  # Shows detailed Claude Code interaction logs
    )
    print(f"Success: {result['success']}")
    print(f"Response: {result['response']}")

    # Verify tool was actually called
    print(f"Tool state: {my_tool.state}")
    
except ConnectionError as e:
    print(f"Connection issue: {e}")
    # Handle Docker, network, or port binding problems
    
except ExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle agent execution or tool failures
```

## Model Selection

Choose the right Claude model for your agent's needs:

### Available Models
- **"haiku"** - Fast and efficient for simple tasks
- **"sonnet"** - Balanced performance (good default choice)
- **"opus"** - Most capable for complex reasoning

### Usage Examples

```python
from claude_agent_toolkit import Agent

# Use fast Haiku model for simple tasks
weather_agent = Agent(
    system_prompt="You are a weather assistant",
    tools=[weather_tool],
    model="haiku"  # Fast, efficient for simple weather queries
)

# Use Sonnet for general-purpose tasks
general_agent = Agent(
    system_prompt="You are a helpful assistant", 
    tools=[calculator_tool, weather_tool],
    model="sonnet"  # Balanced performance
)

# Use Opus for complex analysis
analysis_agent = Agent(
    system_prompt="You are a data analyst",
    tools=[analysis_tool],
    model="opus"  # Maximum reasoning capability
)

# Override model for specific queries
result = await weather_agent.run(
    "Complex weather pattern analysis for next month",
    model="opus"  # Use more capable model for this specific task
)

# Full model IDs also work
agent = Agent(model="claude-3-5-haiku-20241022")
```

### When to Use Each Model
- **Haiku**: Simple queries, basic operations, fast responses needed
- **Sonnet**: General purpose tasks, good balance of speed and capability
- **Opus**: Complex reasoning, detailed analysis, maximum quality needed

## Why Claude Code Agents?

Unlike generic agent frameworks, this toolkit specifically leverages Claude Code's unique capabilities:

1. **Advanced Reasoning** - Use Claude Code's sophisticated decision-making in your agents
2. **Existing Subscription** - Build production agents with your current Claude Code subscription
3. **Stateful Workflows** - Claude Code builds context across multiple tool interactions
4. **Intelligent Orchestration** - Claude Code decides which tools to use and when
5. **Production Infrastructure** - Leverage Claude's robust infrastructure for your agents

### Example: Intelligent Workflow

```python
# Claude Code analyzes data with one tool, then decides to process it with another
# The agent maintains context and makes intelligent decisions about tool usage
# Your tools provide capabilities, Claude Code provides the intelligence
```

## API Reference

### Agent Class

```python
class Agent:
    def __init__(                                          # Initialize agent
        self,
        oauth_token: Optional[str] = None,                 # Your Claude Code token
        system_prompt: Optional[str] = None,               # Custom agent behavior
        tools: Optional[List[BaseTool]] = None,            # Tools to connect automatically
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None  # Model selection
    )
    def connect(self, tool: BaseTool) -> 'Agent'           # Connect custom tools  
    async def run(                                         # Run Claude Code with tools
        self,
        prompt: str,                                       # Instruction for Claude
        verbose: bool = False,                             # Show detailed processing logs
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None  # Override model
    ) -> Dict[str, Any]
```

### BaseTool Class  

```python
class BaseTool:
    def __init__(self)
    def run(self, host="127.0.0.1", port=None, *, workers=None) -> 'BaseTool'
    @property def connection_url(self) -> str
    @property def state(self) -> Any  # Mutable state dictionary
```

### @tool() Decorator

```python
@tool(
    name: Optional[str] = None,           # Tool method name
    description: str = "",               # Method description  
    cpu_bound: bool = False,             # Use process pool
    timeout_s: int = 60,                 # Timeout for CPU-bound operations
    conflict_policy: str = "retry",      # How to handle state conflicts
    max_retries: int = 16                # Max retry attempts
)
```

### Exception Classes

Claude Agent Toolkit provides specific exception types for clear error handling:

```python
# Import exception classes
from claude_agent_toolkit import (
    ClaudeAgentError,     # Base exception for all library errors
    ConfigurationError,   # Missing OAuth tokens, invalid configuration
    ConnectionError,      # Docker, network, port binding failures  
    ExecutionError,       # Agent execution, tool failures, timeouts
    StateError           # Tool lifecycle, state management issues
)

# Exception hierarchy
ClaudeAgentError
├── ConfigurationError    # Configuration issues
├── ConnectionError       # Network/service connectivity
├── ExecutionError       # Runtime execution failures
└── StateError          # State management problems
```

**When to catch each exception:**

- **ConfigurationError**: Handle setup issues, missing tokens, invalid configs
- **ConnectionError**: Handle Docker, network, and port binding failures
- **ExecutionError**: Handle runtime failures, timeouts, tool execution issues  
- **StateError**: Handle tool lifecycle violations, state conflicts
- **ClaudeAgentError**: Catch all library errors with a single handler

## Development Workflow

### 1. Start Docker Desktop
Required for agent execution - must be running before creating Claude Code agents.

### 2. Set OAuth Token  
```bash
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### 3. Create Custom Tools
Inherit from `BaseTool` and implement `@tool` methods that extend Claude Code's capabilities.

### 4. Build Your Agent  
Use the examples in `src/examples/` to see demonstrations or create custom agent scripts.

### 5. Deploy to Production
Use your Claude Code subscription to run agents at scale with custom tool integration.

## Dependencies

### Runtime Dependencies
- `docker>=7.1.0` - Docker container management
- `fastmcp>=2.11.3` - MCP server framework
- `httpx>=0.28.1` - HTTP client for health checks
- `jsonpatch>=1.33` - State management with JSON patches  
- `uvicorn>=0.35.0` - ASGI server for MCP HTTP endpoints

### Docker Environment  
- Python 3.11 with Claude Code SDK
- Node.js 20 with Claude Code CLI
- Non-root user execution for security

## Error Handling

Claude Agent Toolkit uses specific exception types to help you handle errors gracefully:

```python
from claude_agent_toolkit import (
    Agent, BaseTool, tool,
    ClaudeAgentError, ConfigurationError, ConnectionError,
    ExecutionError, StateError
)

# Handle specific error types
try:
    agent = Agent(
        oauth_token="your-token",
        tools=[MyTool().run()]
    )
    result = await agent.run("Process my request")
    
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Handle missing OAuth token, invalid tool config
    
except ConnectionError as e:
    print(f"Connection failed: {e}")
    # Handle Docker, network, port binding issues
    
except ExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle agent execution, tool failures, timeouts
    
except StateError as e:
    print(f"State management issue: {e}")
    # Handle tool lifecycle, state conflicts
    
except ClaudeAgentError as e:
    print(f"Library error: {e}")
    # Catch all library errors
```

## Troubleshooting

### Common Issues

**ConfigurationError: "OAuth token required"**
```python
# Set environment variable
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Or pass directly to Agent
agent = Agent(oauth_token='your-token-here')
```

**ConnectionError: "Cannot connect to Docker"**
- Ensure Docker Desktop is running
- Check Docker daemon is accessible
- Linux: `sudo systemctl start docker`

**ConnectionError: "Port binding failed"**
```python
# Let tools auto-select available ports
tool = MyTool().run()  # Auto-selects port

# Or specify different port
tool = MyTool().run(port=9000)
```

**StateError: "Tool is not running"**
```python
# Start tool before accessing properties
tool = MyTool()
tool.run()  # Start the tool
url = tool.connection_url  # Now accessible
```

**ExecutionError: "Operation timed out"**
```python
# Increase timeout for CPU-bound operations
@tool(cpu_bound=True, timeout_s=300)  # 5 minute timeout
async def heavy_computation(self):
    pass
```

### Debug Mode
```python
from claude_agent_toolkit import set_logging, LogLevel

# Enable detailed logging
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Run with verbose output
result = await agent.run("your prompt", verbose=True)
```

## Contributing

1. Create custom tools for different Claude Code agent use cases
2. Add new agent development patterns and templates
3. Improve Docker image efficiency and security
4. Enhance state management and conflict resolution
5. Add support for additional MCP server types

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [Claude Code](https://claude.ai/code) - Official Claude Code interface (required for this framework)
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) - Protocol for AI-tool integration
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast MCP server implementation