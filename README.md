# Claude Code Agent Development Kit (claude-adk)

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

- **Agent Framework** (`src/claude_adk/agent/`) - Docker-isolated Agent class that runs Claude Code with MCP tool support
- **MCP Tool Framework** (`src/claude_adk/tool/`) - BaseTool class for creating custom MCP tools with state management
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
pip install claude-adk

# Using uv
uv add claude-adk

# Using poetry
poetry add claude-adk

# Set your OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### Run the Demo

```bash
# Clone the repository for examples
git clone https://github.com/cheolwanpark/claude-adk.git
cd claude-adk

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
from claude_adk import BaseTool, tool

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
from claude_adk import Agent

# Create and start tool
my_tool = MyTool().run(workers=2)

# Create agent and connect to tool  
agent = Agent()
agent.connect(my_tool)

# Run agent with prompt
result = await agent.run("Please increment the counter twice and tell me the result")
print(f"Success: {result['success']}")
print(f"Response: {result['response']}")

# Verify tool was actually called
print(f"Tool state: {my_tool.state}")
```

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
    def __init__(self, oauth_token: Optional[str] = None)  # Your Claude Code token
    def connect(self, tool: BaseTool) -> 'Agent'           # Connect custom tools  
    async def run(self, prompt: str) -> Dict[str, Any]     # Run Claude Code with tools
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

## Troubleshooting

### Common Issues

**"Cannot connect to Docker"**
- Ensure Docker Desktop is running
- Check Docker daemon is accessible

**"OAuth token required"**  
- Set `CLAUDE_CODE_OAUTH_TOKEN` environment variable
- Get token from [Claude Code](https://claude.ai/code)

**Tool connection failures**
- Check tool health endpoints are accessible
- Verify port conflicts (tools auto-assign ports)
- Review Docker network connectivity

### Debug Mode
```bash
# Enable detailed logging
export CLAUDE_DEBUG=1
python main.py
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