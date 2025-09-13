# Claude Agent Toolkit Examples

This directory contains practical examples demonstrating the Claude Agent Toolkit capabilities. Each example is a complete, standalone project showing different aspects of building Claude Code agents with custom tools.

## Prerequisites

1. **Claude Code OAuth Token** - Get your token from [claude.ai/code](https://claude.ai/code)
2. **Python 3.12+** with uv package manager
3. **Docker Desktop** - Required for most examples (except subprocess)
4. **Internet Connection** - For weather example and MCP external servers

## Setup

### 1. Set Your OAuth Token
```bash
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### 2. Install Dependencies
```bash
# From the examples directory
uv sync
```

### 3. Start Docker Desktop
Ensure Docker Desktop is running (except for subprocess example).

## How to Run Examples

Each example can be run in two modes:

- **Demo Mode** (default): Runs predefined scenarios
- **Interactive Mode**: Allows user input for experimentation

```bash
# Demo mode
cd example_name && python main.py

# Interactive mode
cd example_name && python main.py --interactive
```

## Examples

### Calculator (`calculator/`)

**Mathematical assistant with state management and operation history**

```bash
cd calculator
python main.py                 # Demo: arithmetic, advanced operations, history
python main.py --interactive   # Interactive calculator mode
```

**What it demonstrates:**
- Custom tool implementation with `@tool` decorator
- State management across operations (history, last result)
- Multi-step mathematical problem solving
- Professional agent prompting for mathematical explanations

**Key features:**
- Basic operations: addition, subtraction, multiplication, division
- Advanced operations: power, square root
- Operation history tracking
- Last result persistence

---

### Weather (`weather/`)

**Real-time weather assistant with API integration and favorites**

```bash
cd weather
python main.py                 # Demo: current conditions, forecasts, comparisons
python main.py --interactive   # Interactive weather assistant
```

**What it demonstrates:**
- External API integration (wttr.in weather service)
- Async operations and data processing
- State management (favorites, query history)
- Real-world data interpretation and advice

**Key features:**
- Current weather conditions for any location
- Multi-day weather forecasts
- Weather comparisons between locations
- Favorite locations management
- Travel planning advice

---

### Subprocess (`subprocess/`)

**Docker-free execution with MessageReflector tool**

```bash
cd subprocess
python main.py                 # Demo: message reflection, tool status
python main.py --interactive   # Interactive subprocess mode
```

**What it demonstrates:**
- SubprocessExecutor (6x faster startup, no Docker required)
- Direct subprocess execution without containers
- MessageReflector tool implementation
- MCP tool integration patterns

**Key features:**
- No Docker dependency
- ~0.5s startup vs ~3s for Docker
- Message reflection with metadata
- Tool status and history tracking

---

### MCP External Servers (`mcp/`)

**Integration with external MCP servers using multiple transports**

```bash
cd mcp
python main.py                 # Demo: stdio and HTTP transports
```

**What it demonstrates:**
- External MCP server integration
- Multiple transport types (stdio, HTTP)
- StdioMCPTool and HttpMCPTool usage
- Real-world MCP ecosystem integration

**Key features:**
- Stdio transport: Direct command execution
- HTTP transport: REST-style HTTP endpoints
- Everything server integration (npx @modelcontextprotocol/server-everything)
- Mixed internal/external tool scenarios

---

### FileSystem (`filesystem/`)

**File operations with permission-based security**

```bash
cd filesystem
python main.py                 # Demo: read/write operations with permissions
```

**What it demonstrates:**
- Built-in FileSystemTool usage
- Permission-based file access control
- Agent-driven file operations
- Validation of agent actions

**Key features:**
- Read/write file operations
- Permission pattern matching
- Safe file system access
- Agent instruction validation

---

### DataTransfer (`datatransfer/`)

**Type-safe data transfer with Pydantic models**

```bash
cd datatransfer
python main.py                 # Demo: single models, multiple models, complex data
```

**What it demonstrates:**
- Built-in DataTransferTool usage
- Type-safe data validation and transfer
- Pydantic BaseModel integration
- Dynamic tool creation patterns

**Key features:**
- Generic tool for any Pydantic BaseModel
- Automatic schema inclusion in descriptions
- Nested model support
- Multiple tools in single agent

## Common Patterns

### Error Handling
All examples include comprehensive error handling:
```python
try:
    response = await agent.run("Your prompt")
except ConfigurationError as e:
    print(f"❌ Configuration Error: {e}")
except ConnectionError as e:
    print(f"❌ Connection Error: {e}")
except ExecutionError as e:
    print(f"❌ Execution Error: {e}")
```

### Agent Creation
```python
# Method 1: New pattern (recommended)
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    tools=[your_tool],
    model="haiku"  # Optional: haiku, sonnet, opus
)

# Method 2: Legacy pattern (still works)
agent = Agent()
agent.connect(your_tool)
```

### Tool Implementation
```python
from claude_agent_toolkit import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.data = {}  # Explicit data management

    @tool(description="Your tool description")
    async def my_method(self, param: str) -> dict:
        return {"result": f"processed_{param}"}
```

## Troubleshooting

### Common Issues

**"CLAUDE_CODE_OAUTH_TOKEN not set"**
- Solution: `export CLAUDE_CODE_OAUTH_TOKEN='your-token'`
- Get token from: [claude.ai/code](https://claude.ai/code)

**"Cannot connect to Docker"**
- Solution: Start Docker Desktop and retry
- Alternative: Use subprocess example (no Docker needed)

**"Import claude_agent_toolkit could not be resolved"**
- Solution: Run `uv sync` from the examples directory

**Weather API timeout**
- Solution: Check internet connection; wttr.in may be temporarily unavailable

**Port conflicts**
- Solution: Tools auto-select available ports; wait a moment and retry

### Debug Mode
Enable verbose logging for troubleshooting:
```python
# In any example
response = await agent.run("prompt", verbose=True)
```

## Next Steps

These examples serve as templates for building your own Claude Code agents:

1. **Copy example structure** as a starting point
2. **Implement your tools** following the BaseTool pattern
3. **Create appropriate prompts** for your domain
4. **Add state management** as needed for your use case
5. **Test thoroughly** with both demo and interactive modes

Each example is designed to be educational, practical, and extensible for real-world applications.