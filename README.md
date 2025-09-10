# Claude Code Agent Toolkit (claude-agent-toolkit)

A Python framework for building Claude Code agents with custom tools, designed to leverage Claude Code's advanced reasoning capabilities with your subscription token. The framework provides Docker-isolated environments where Claude Code can orchestrate custom MCP tools for production workflows.

## Key Features

- **Claude Code Integration** - Leverage Claude Code's advanced reasoning with your existing subscription token
- **Flexible Execution Modes** - Choose between Docker isolation or direct subprocess execution
- **Explicit Data Management** - Users control their own data without automatic state management  
- **CPU-bound Operations** - Support for CPU-intensive operations with process pools and parallel execution
- **Multi-tool Coordination** - Claude Code orchestrates multiple tools in complex workflows
- **Production Ready** - Build scalable agents using Claude Code's capabilities with custom tool integration

## Architecture

### Core Components

- **Agent Framework** - Main Agent class supporting Docker or subprocess execution with MCP tool integration
- **MCP Tool Framework** - BaseTool class for creating custom tools with explicit data management
- **Example Tools** - Demonstration tools showing practical agent development patterns
- **Execution Modes** - Docker isolation or direct subprocess execution for different use cases

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

# Start Docker Desktop first, then run the Docker examples
# Calculator example (Docker executor):
cd src/examples/calculator && python main.py
# Weather example (Docker executor):
cd src/examples/weather && python main.py
# Subprocess example (no Docker required):
cd src/examples/subprocess && python main.py
# Filesystem example (with FileSystemTool):
cd src/examples/filesystem && python main.py
# DataTransfer example (with DataTransferTool):
cd src/examples/datatransfer && python main.py
```

This will run demonstration examples:
1. **Calculator Demo** - Shows stateful operations, parallel processing (factorial, fibonacci, prime checking), and mathematical problem solving
2. **Weather Demo** - Demonstrates external API integration with real-time data and async operations
3. **Subprocess Demo** - Shows subprocess executor usage without Docker dependency
4. **Filesystem Demo** - Demonstrates FileSystemTool with pattern-based permissions and agent integration
5. **DataTransfer Demo** - Shows DataTransferTool with type-safe data transfer, validation, and multi-model scenarios

### Executor Options

- **Docker** (default): Isolated environment, requires Docker Desktop
- **Subprocess**: Direct execution, faster startup (~0.5s vs ~3s)

Choose based on your needs. Docker provides isolation for local testing, subprocess works well in clean production environments.

```python
from claude_agent_toolkit import Agent, ExecutorType

# Default: Docker executor
agent = Agent(tools=[my_tool])

# Alternative: Subprocess executor  
agent = Agent(tools=[my_tool], executor=ExecutorType.SUBPROCESS)
```

## Tool Development

### Creating Custom Tools

Create tools by inheriting from `BaseTool` and using the `@tool()` decorator:

```python
from claude_agent_toolkit import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()  # Server starts automatically
        # Explicit data management - no automatic state management
        self.counter = 0
        self.operations = []
    
    @tool(description="Increment counter and return new value")
    async def increment(self) -> dict:
        self.counter += 1
        return {"value": self.counter}
    
    @tool(description="Heavy computation with parallel processing", parallel=True, timeout_s=120)  
    def compute_heavy(self, data: str) -> dict:
        # CPU-intensive operation runs under ProcessPoolExecutor
        # Note: ProcessPoolExecutor creates new instance, self.counter won't persist
        import time
        time.sleep(2)  # Simulate heavy computation
        return {"processed": f"Heavy result for {data}", "parallel_execution": True}
```

### Context Manager Support

For explicit resource management, use the context manager pattern:

```python
# Single tool with guaranteed cleanup
with MyTool(workers=2) as tool:
    agent = Agent(tools=[tool], executor=ExecutorType.SUBPROCESS)
    result = await agent.run("Process my data")
    print(f"Result: {result}")
# Server automatically cleaned up here

# Multiple tools in one statement
with MyTool() as calc_tool, WeatherTool() as weather_tool:
    agent = Agent(tools=[calc_tool, weather_tool])
    result = await agent.run("Calculate something and check weather")
    print(f"Result: {result}")
# Both tools cleaned up automatically

# Parameters can be passed to constructor
with MyTool(host="127.0.0.1", port=8080, workers=4, log_level="INFO") as tool:
    # Tool server starts immediately with specified configuration
    agent = Agent(tools=[tool])
    result = await agent.run("Heavy computation task", verbose=True)  # Prints to console
    print(f"Task result: {result}")
# Guaranteed cleanup even if exceptions occur
```

## Built-In Tools

Claude Agent Toolkit includes ready-to-use tools that extend your agents' capabilities:

### FileSystemTool

Secure filesystem access with pattern-based permissions - control exactly what files your agent can read, write, or modify.

**Key Features:**
- Pattern-based permissions using glob patterns (`*.txt`, `data/**`, `logs/*.log`)
- Four operations: `list()`, `read()`, `write()`, `update()`
- Security: Path traversal prevention and root directory containment
- Conflict resolution: Write permission takes precedence over read

**Basic Usage:**

```python
from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import FileSystemTool

# Define permission rules
permissions = [
    ("*.txt", "read"),          # Read all text files
    ("data/**", "write"),       # Write access to data directory
    ("logs/*.log", "read"),     # Read-only log files
    ("secrets/*", "read"),      # Read secrets directory
]

# Create filesystem tool with permissions
fs_tool = FileSystemTool(
    permissions=permissions,
    root_dir="/path/to/workspace"  # Restrict to specific directory
)

# Use with an agent
agent = Agent(
    system_prompt="You are a file manager assistant",
    tools=[fs_tool]
)

result = await agent.run(
    "List all text files, read the latest log, and create a summary in data/report.txt"
)
```

**Permission Patterns:**
- `*.extension` - Match files by extension anywhere
- `dir/*` - Match files directly in a directory
- `dir/**` - Match all files recursively in a directory
- `specific/file.txt` - Match a specific file

**Common Use Cases:**
- **Log Analysis**: Read-only access to log files for monitoring and analysis
- **Data Processing**: Read input files, write results to specific directories
- **Configuration Management**: Controlled access to config files
- **Report Generation**: Read from multiple sources, write reports to designated locations

**Security Notes:**
- All paths are confined to the specified `root_dir`
- Path traversal attempts (`../`) are blocked
- When permissions conflict, write permission wins (write includes read)

### DataTransferTool

Generic tool for transferring structured data between Claude agents and host applications using any Pydantic BaseModel.

**Key Features:**
- Generic implementation works with any Pydantic BaseModel subclass
- Automatic schema inclusion in tool descriptions for Claude
- Type-safe data validation and transfer with runtime validation
- Dynamic class creation for distinct tool identities
- Simple transfer/get interface for host applications

**Basic Usage:**

```python
from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import DataTransferTool
from pydantic import BaseModel, Field

# Define your data model
class UserProfile(BaseModel):
    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    email: str = Field(..., description="Email address")
    interests: List[str] = Field(default_factory=list, description="User interests")

# Create tool for specific model with distinct name
user_tool = DataTransferTool.create(UserProfile, "UserProfileTool")

# Use with an agent
agent = Agent(
    system_prompt="You are a data assistant for user profile transfers.",
    tools=[user_tool]
)

# Transfer data through Claude
result = await agent.run(
    "Transfer user data: name='Alice Johnson', age=28, email='alice@example.com', "
    "interests=['programming', 'hiking']"
)

# Retrieve validated data from host
user_data = user_tool.get()
if user_data:
    print(f"Retrieved: {user_data.name}, age {user_data.age}")
```

**Advanced Data Types:**
- **Nested Models**: Transfer complex data with embedded objects
- **Lists of Models**: Handle arrays of structured data
- **Dictionary Models**: Transfer data with model values in dictionaries
- **Field Constraints**: Automatic validation with Pydantic constraints

**Common Use Cases:**
- **Form Data Transfer**: Collect and validate user input through conversational interface
- **API Data Exchange**: Transfer structured data between systems with validation
- **Configuration Transfer**: Pass complex settings with type safety
- **Data Pipeline**: Move validated data between processing stages
- **Multi-Model Workflows**: Use different tools for different data types in same session

**Factory Method:**
```python
# Create tools for different models with distinct identities
user_tool = DataTransferTool.create(UserProfile, "UserTool")
product_tool = DataTransferTool.create(ProductInfo, "ProductTool")
order_tool = DataTransferTool.create(Order, "OrderTool")  # Nested models

# Each appears as a completely different tool to Claude
agent = Agent(tools=[user_tool, product_tool, order_tool])
```

**Host Integration:**
```python
# Simple retrieval interface
user_data = user_tool.get()              # Get transferred data
has_data = user_tool.has_data()          # Check if data exists
user_tool.clear()                        # Clear stored data
schema = user_tool.get_schema()          # Get JSON schema
json_data = user_tool.to_json()          # Get as JSON string
```

### Using Tools with Agents

```python
from claude_agent_toolkit import Agent, ExecutorType, ConnectionError, ExecutionError

try:
    # Create tool (server starts automatically)
    my_tool = MyTool(workers=2)

    # Create agent with tools
    agent = Agent(
        system_prompt="You are a helpful assistant specialized in calculations",
        tools=[my_tool],
        executor=ExecutorType.DOCKER  # Optional - Docker is default
    )

    # Run agent with prompt (verbose=True prints detailed output to console)
    result = await agent.run(
        "Please increment the counter twice and tell me the result",
        verbose=True  # Prints detailed execution info to console
    )
    print(f"Response: {result}")  # result is the string response
    
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
from claude_agent_toolkit import Agent, ExecutorType

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
    model="sonnet",  # Balanced performance
    executor=ExecutorType.SUBPROCESS  # Use subprocess for this example
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
print(f"Analysis result: {result}")

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
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None,  # Model selection
        executor: Optional[ExecutorType] = None            # Executor type (Docker/Subprocess)
    )
    def connect(self, tool: BaseTool) -> 'Agent'           # Connect custom tools  
    async def run(                                         # Run Claude Code with tools
        self,
        prompt: str,                                       # Instruction for Claude
        verbose: bool = False,                             # Print detailed output to console
        model: Optional[Union[Literal["opus", "sonnet", "haiku"], str]] = None  # Override model
    ) -> str                                               # Returns Claude's response as string
```

### BaseTool Class  

```python
class BaseTool:
    def __init__(self, host="127.0.0.1", port=None, *, workers=None, log_level="ERROR")
    @property def connection_url(self) -> str  # Always accessible after construction
    @property def health_url(self) -> str      # Always accessible after construction
    def __enter__(self) -> 'BaseTool'          # Context manager support
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool
    def __del__(self)                          # Automatic cleanup on destruction
```

### @tool() Decorator

```python
@tool(
    name: Optional[str] = None,           # Tool method name
    description: str = "",               # Method description  
    parallel: bool = False,              # Use process pool
    timeout_s: int = 60,                 # Timeout for parallel operations
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
)

# Exception hierarchy
ClaudeAgentError
├── ConfigurationError    # Configuration issues
├── ConnectionError       # Network/service connectivity
└── ExecutionError       # Runtime execution failures
```

**When to catch each exception:**

- **ConfigurationError**: Handle setup issues, missing tokens, invalid configs
- **ConnectionError**: Handle Docker, network, and port binding failures
- **ExecutionError**: Handle runtime failures, timeouts, tool execution issues
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
  
- `uvicorn>=0.35.0` - ASGI server for MCP HTTP endpoints

### Docker Environment  
- Python 3.11 with Claude Code SDK
- Node.js 20 with Claude Code CLI
- Non-root user execution for security

## Error Handling

Claude Agent Toolkit uses specific exception types to help you handle errors gracefully:

```python
from claude_agent_toolkit import (
    Agent, BaseTool, tool, ExecutorType,
    ClaudeAgentError, ConfigurationError, ConnectionError,
    ExecutionError
)

# Handle specific error types
try:
    agent = Agent(
        oauth_token="your-token",
        tools=[MyTool()],
        executor=ExecutorType.SUBPROCESS
    )
    result = await agent.run("Process my request")
    print(f"Result: {result}")
    
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Handle missing OAuth token, invalid tool config
    
except ConnectionError as e:
    print(f"Connection failed: {e}")
    # Handle Docker, network, port binding issues
    
except ExecutionError as e:
    print(f"Execution failed: {e}")
    # Handle agent execution, tool failures, timeouts
    
    
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
tool = MyTool()  # Auto-selects port

# Or specify different port
tool = MyTool(port=9000)
```

**ConnectionError: "Tool server failed to start"**
```python
# Tool server starts automatically in constructor
tool = MyTool()  # Server starts immediately
url = tool.connection_url  # Always accessible after construction
```

**ExecutionError: "Operation timed out"**
```python
# Increase timeout for parallel operations
@tool(parallel=True, timeout_s=300)  # 5 minute timeout
def heavy_computation(self, data: str) -> dict:
    # Parallel operations must be sync functions
    return {"result": "processed"}
```

### Debug Mode
```python
from claude_agent_toolkit import set_logging, LogLevel

# Enable detailed logging
set_logging(LogLevel.DEBUG, show_time=True, show_level=True)

# Run with verbose output (prints to console)
result = await agent.run("your prompt", verbose=True)
print(f"Response: {result}")
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