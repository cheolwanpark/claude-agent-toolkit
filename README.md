# Claude Agent Development Kit (claude-adk)

A testing framework for verifying that Claude Code agents actually call MCP tools correctly, rather than hallucinating tool interactions. The framework provides Docker-isolated environments where agents can connect to custom MCP tools and verify tool interactions through unpredictable results.

## Key Features

- **Verification Through Unpredictability** - Uses tools that generate unpredictable results (secrets, timestamped hashes) to verify actual tool calls vs hallucinations
- **Docker Isolation** - Complete isolation of agent execution environment with Claude Code CLI
- **Advanced State Management** - JSON patch-based state management with conflict resolution and automatic retries  
- **CPU-bound Operations** - Support for CPU-intensive operations with process pools and parallel execution
- **Multi-tool Coordination** - Test agents' ability to coordinate multiple tools in complex workflows
- **Runtime Verification** - No traditional unit tests - verification happens through runtime agent testing

## Architecture

### Core Components

- **Agent Framework** (`agent.py`) - Docker-isolated Agent class that runs Claude Code with MCP tool support
- **MCP Tool Framework** (`tool.py`) - BaseTool class for creating custom MCP tools with state management
- **Demo Tools** (`main.py`) - SecretGeneratorTool and HashComputerTool for verification testing
- **Docker Environment** (`Dockerfile`) - Isolated environment with Claude Code CLI and dependencies  
- **Entry Point** (`entrypoint.py`) - Runs inside Docker containers to execute Claude Code queries

## Quick Start

### Prerequisites

- **Python 3.12+** with `uv` package manager
- **Docker Desktop** (must be running)
- **Claude Code OAuth Token** - Get from [Claude Code](https://claude.ai/code)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd claude-adk

# Install dependencies
uv sync

# Set your OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### Run the Demo

```bash
# Start Docker Desktop first, then run the verification demo
python main.py
```

This will run three verification tests:
1. **Secret Generator** - Verify agent generates and retrieves secrets
2. **Hash Computer** - Verify agent computes timestamped hashes  
3. **Multi-Tool** - Verify agent coordinates multiple tools

## Tool Development

### Creating Custom Tools

Create tools by inheriting from `BaseTool` and using the `@tool()` decorator:

```python
from tool import BaseTool, tool

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
from agent import Agent

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

## Verification Approach

The framework verifies agent behavior by:

1. **Unpredictable Results** - Tools generate results that cannot be predicted (random secrets, timestamps)
2. **State Inspection** - Check tool internal state to confirm methods were actually called
3. **Response Analysis** - Verify unpredictable values appear in agent responses
4. **Multi-tool Testing** - Ensure agents can coordinate multiple tools correctly

### Example: Secret Verification

```python
# Agent claims to have generated secret "abc123"
# But tool state shows actual secret is "x7y9z2k1" 
# ’ Agent is hallucinating, not calling the tool
```

## API Reference

### Agent Class

```python
class Agent:
    def __init__(self, oauth_token: Optional[str] = None)
    def connect(self, tool: BaseTool) -> 'Agent'
    async def run(self, prompt: str) -> Dict[str, Any]
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
Required for agent execution - must be running before running any tests.

### 2. Set OAuth Token  
```bash
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### 3. Create Custom Tools
Inherit from `BaseTool` and implement `@tool` methods with unpredictable behavior.

### 4. Run Verification  
Use `python main.py` to test agent-tool interactions or create custom test scripts.

### 5. Check Tool State
Always verify tools were actually called by checking their internal state.

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

1. Create custom tools for different verification scenarios
2. Add new verification patterns and edge cases  
3. Improve Docker image efficiency and security
4. Enhance state management and conflict resolution
5. Add support for additional MCP server types

## License

[Add your license here]

## Related Projects

- [Claude Code](https://claude.ai/code) - Official Claude Code interface
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) - Protocol for AI-tool integration
- [FastMCP](https://github.com/jlowin/fastmcp) - Fast MCP server implementation