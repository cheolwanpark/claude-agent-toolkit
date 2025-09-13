# Claude Agent Toolkit

**Simplified AI agent development with decorator-based tools and external MCP server integration**

A Python framework that solves the complexity of claude-code-sdk tool integration through an intuitive decorator-based approach. Created to provide a stable, consistent development experience similar to Google's Agent Development Kit (ADK), this toolkit enables effortless creation of Claude Code agents with custom MCP tools and seamless integration with existing MCP servers.

## Table of Contents

- [Why Claude Agent Toolkit?](#why-claude-agent-toolkit)
- [When Should You Use This?](#when-should-you-use-this)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Usage Examples](#usage-examples)
- [Core Features](#core-features)
- [Architecture](#architecture)
- [Built-in Tools](#built-in-tools)
- [Creating Custom Tools](#creating-custom-tools)
- [FAQ](#faq)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Why Claude Agent Toolkit?

### The Problem
Working directly with claude-code-sdk presents two major challenges:
1. **Complex Tool Integration** - Manual MCP server setup, connection handling, and tool registration
2. **Inconsistent Execution** - Different local environments causing unpredictable behavior

### The Solution
Claude Agent Toolkit solves these issues through:
- **🎯 Decorator-Based Tools** - Simple `@tool` decorator converts any Python function into a Claude-compatible tool
- **🐳 Isolated Execution** - Docker containers ensure consistent behavior across all environments
- **⚡ Zero Configuration** - Automatic MCP server management and tool discovery

*"An intuitive and stable development experience similar to Google's ADK"* - [Cheolwan Park, Creator](https://blog.codingvillain.com/post/claude-agent-toolkit)

### Before vs After

**Before (Direct claude-code-sdk):**
```python
# Complex MCP server setup
# Manual connection management
# Environment-specific configurations
# Inconsistent tool behavior
```

**After (Claude Agent Toolkit):**
```python
class CalculatorTool(BaseTool):
    @tool(description="Adds two numbers together")
    async def add(self, a: float, b: float) -> dict:
        return {"result": a + b}

agent = Agent(tools=[CalculatorTool()])
```

## When Should You Use This?

> **Update**: Recent updates to claude-code-sdk have simplified tool addition significantly. This has substantially reduced the necessity for Claude Agent Toolkit compared to when it was first created.

### Use claude-code-sdk When:
- Building simple agents with basic tooling needs
- Rapid prototyping and experimentation
- Utilizing local environment configurations
- Working with subprocess execution only

### Use Claude Agent Toolkit When:
- **Isolated Execution Required** - Docker containers for production consistency
- **CPU-Intensive Tools** - Parallel execution in separate process pools with `parallel=True`
- **Production Environments** - Consistent execution across different deployments
- **Complex Tool Orchestration** - Multiple tools with state management

### Quick Comparison

| Feature | claude-code-sdk | Claude Agent Toolkit |
|---------|----------------|---------------------|
| **Setup Complexity** | Simple (recent updates) | Minimal with decorators |
| **Execution** | Same process | Docker/Subprocess isolation |
| **CPU-Intensive Tasks** | Blocking | Parallel with `parallel=True` |
| **Environment Consistency** | Variable | Guaranteed with Docker |
| **Best For** | Prototyping, simple tools | Production, complex tools |

## Quick Start

```python
from claude_agent_toolkit import Agent, BaseTool, tool

# 1. Create a custom tool with @tool decorator
class CalculatorTool(BaseTool):
    @tool(description="Adds two numbers together")
    async def add(self, a: float, b: float) -> dict:
        result = a + b
        return {
            "operation": f"{a} + {b}",
            "result": result,
            "message": f"The result of adding {a} and {b} is {result}"
        }

# 2. Create and run an agent
async def main():
    agent = Agent(
        system_prompt="You are a helpful calculator assistant",
        tools=[CalculatorTool()],
        model="sonnet"  # haiku, sonnet, or opus
    )

    result = await agent.run("What is 15 + 27?")
    print(result)  # Claude will use your tool and return the answer

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Installation & Setup

### Prerequisites

- **Python 3.12+** with `uv` package manager
- **Docker Desktop** (required for Docker executor, recommended)
- **Claude Code OAuth Token** - Get from [Claude Code](https://claude.ai/code)

### Install the Package

```bash
# Using pip
pip install claude-agent-toolkit

# Using uv (recommended)
uv add claude-agent-toolkit

# Using poetry
poetry add claude-agent-toolkit
```

### Set Your OAuth Token

Get your token by running `claude setup-token` in your terminal, then:

```bash
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
```

### Quick Verification

```bash
# Clone examples (optional)
git clone https://github.com/cheolwanpark/claude-agent-toolkit.git
cd claude-agent-toolkit/src/examples/calculator
python main.py
```

## Usage Examples

### Basic Agent with Custom Tool

```python
from claude_agent_toolkit import Agent, BaseTool, tool, ExecutorType

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.counter = 0  # Explicit data management

    @tool(description="Increment counter and return value")
    async def increment(self) -> dict:
        self.counter += 1
        return {"value": self.counter}

# Docker executor (default, production-ready)
agent = Agent(tools=[MyTool()])

# Subprocess executor (faster startup, development)
agent = Agent(tools=[MyTool()], executor=ExecutorType.SUBPROCESS)

result = await agent.run("Increment the counter twice")
```

### Model Selection

```python
# Fast and efficient for simple tasks
weather_agent = Agent(
    tools=[weather_tool],
    model="haiku"
)

# Balanced performance (default)
general_agent = Agent(
    tools=[calc_tool, weather_tool],
    model="sonnet"
)

# Most capable for complex reasoning
analysis_agent = Agent(
    tools=[analysis_tool],
    model="opus"
)

# Override per query
result = await weather_agent.run(
    "Complex weather pattern analysis",
    model="opus"
)
```

### CPU-Intensive Operations

```python
class HeavyComputeTool(BaseTool):
    @tool(description="Heavy computation", parallel=True, timeout_s=120)
    def process_data(self, data: str) -> dict:
        # Sync function - runs in separate process
        import time
        time.sleep(5)  # Simulate heavy work
        return {"processed": f"result_{data}"}
```

### Error Handling

```python
from claude_agent_toolkit import (
    Agent, BaseTool,
    ConfigurationError, ConnectionError, ExecutionError
)

try:
    agent = Agent(tools=[MyTool()])
    result = await agent.run("Process my request")

except ConfigurationError as e:
    print(f"Setup issue: {e}")  # Missing token, invalid config

except ConnectionError as e:
    print(f"Connection failed: {e}")  # Docker, network issues

except ExecutionError as e:
    print(f"Execution failed: {e}")  # Tool failures, timeouts
```

## Core Features

- **🎯 Decorator-Based Tools** - Transform any Python function into a Claude tool with simple `@tool` decorator
- **🔌 External MCP Integration** - Connect to existing MCP servers via stdio and HTTP transports
- **🐳 Isolated Execution** - Docker containers ensure consistent behavior across all environments
- **⚡ Zero Configuration** - Automatic MCP server management, port selection, and tool discovery
- **🔧 Flexible Execution Modes** - Choose Docker isolation (production) or subprocess (development)
- **📝 Explicit Data Management** - You control data persistence with no hidden state
- **⚙️ CPU-bound Operations** - Process pools for heavy computations with parallel processing
- **🎭 Multi-tool Coordination** - Claude Code intelligently orchestrates multiple tools
- **🏗️ Production Ready** - Built for scalable, reliable agent deployment

## Architecture

### Execution Modes

| Feature | Docker (Default) | Subprocess |
|---------|------------------|------------|
| **Isolation** | Full container isolation | Process isolation only |
| **Setup Time** | ~3 seconds | ~0.5 seconds |
| **Use Case** | Production, testing | Development, CI/CD |
| **Requirements** | Docker Desktop | None |

### Component Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Tools    │    │      Agent       │    │  Claude Code    │
│  (MCP Servers)  │◄──►│   (Orchestrator) │◄──►│   (Reasoning)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Host Process  │    │ Docker Container │    │ Claude Code API │
│   (localhost)   │    │   or Subprocess  │    │   (claude.ai)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## External MCP Server Integration

### Connect to Existing MCP Servers

Integrate with any existing MCP server using stdio or HTTP transport:

```python
from claude_agent_toolkit import Agent
from claude_agent_toolkit.tool.mcp import StdioMCPTool, HttpMCPTool

# Connect to an MCP server via command execution
everything_server = StdioMCPTool(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-everything"],
    name="everything"
)

# Connect to an HTTP MCP server
http_server = HttpMCPTool(
    url="http://localhost:3001/mcp",
    name="my-http-server"
)

# Mix with your custom tools
agent = Agent(
    system_prompt="You can use both custom and external tools",
    tools=[MyCustomTool(), everything_server, http_server]
)

result = await agent.run("Use the everything server to echo 'Hello World'")
```

### When to Use External MCP Integration

- **Existing MCP Ecosystem**: Leverage community MCP servers and tools
- **Language Diversity**: Use MCP servers written in Node.js, Python, Go, etc.
- **Specialized Tools**: Integrate domain-specific tools without reimplementation
- **Rapid Prototyping**: Quickly test MCP servers before building custom equivalents
- **Service Architecture**: Connect to MCP servers running as microservices

### Supported Transports

| Transport | Use Case | Example |
|-----------|----------|---------|
| **Stdio** | Command-line tools, npm packages | `npx` servers, Python scripts |
| **HTTP** | Web services, microservices | REST APIs, containerized servers |

## Built-in Tools

### FileSystemTool - Secure File Operations

Control exactly what files your agent can access with pattern-based permissions.

```python
from claude_agent_toolkit.tools import FileSystemTool

# Define access patterns
permissions = [
    ("*.txt", "read"),          # Read all text files
    ("data/**", "write"),       # Write to data directory
    ("logs/*.log", "read"),     # Read log files only
]

fs_tool = FileSystemTool(
    permissions=permissions,
    root_dir="/path/to/workspace"  # Restrict to directory
)

agent = Agent(
    system_prompt="You are a file manager assistant",
    tools=[fs_tool]
)

result = await agent.run(
    "List all text files and create a summary in data/report.txt"
)
```

### DataTransferTool - Type-Safe Data Transfer

Transfer structured data between Claude agents and your application using Pydantic models.

```python
from claude_agent_toolkit.tools import DataTransferTool
from pydantic import BaseModel, Field
from typing import List

class UserProfile(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    interests: List[str] = Field(default_factory=list)

# Create tool for specific model
user_tool = DataTransferTool.create(UserProfile, "UserProfileTool")

agent = Agent(
    system_prompt="You handle user profile data transfers",
    tools=[user_tool]
)

# Transfer data through Claude
await agent.run(
    "Transfer user: Alice Johnson, age 28, interests programming and hiking"
)

# Retrieve validated data
user_data = user_tool.get()
if user_data:
    print(f"Retrieved: {user_data.name}, age {user_data.age}")
```

## Creating Custom Tools

### Basic Tool Pattern

```python
from claude_agent_toolkit import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()  # Server starts automatically
        self.data = {}      # Explicit data management

    @tool(description="Async operation")
    async def process_async(self, data: str) -> dict:
        # Async operations for I/O, API calls
        return {"result": f"processed_{data}"}

    @tool(description="CPU-intensive operation", parallel=True, timeout_s=60)
    def process_heavy(self, data: str) -> dict:
        # Sync function - runs in separate process
        # Note: New instance created, self.data won't persist
        import time
        time.sleep(2)
        return {"result": f"heavy_{data}"}
```

### Context Manager Support

```python
# Single tool with guaranteed cleanup
with MyTool() as tool:
    agent = Agent(tools=[tool])
    result = await agent.run("Process my data")
# Server automatically cleaned up

# Multiple tools
with MyTool() as calc_tool, WeatherTool() as weather_tool:
    agent = Agent(tools=[calc_tool, weather_tool])
    result = await agent.run("Calculate and check weather")
# Both tools cleaned up automatically
```

## FAQ

### What is Claude Agent Toolkit?

A Python framework that lets you build AI agents using Claude Code with custom tools. Unlike generic agent frameworks, this specifically leverages Claude Code's advanced reasoning capabilities with your existing subscription.

### How is this different from other agent frameworks?

- **Uses Claude Code**: Leverages Claude's production infrastructure and reasoning
- **MCP Protocol**: Industry-standard tool integration, not proprietary APIs
- **Explicit Data**: You control data persistence, no hidden state management
- **Production Focus**: Built for real deployment, not just experiments

### Do I need Docker?

Docker is recommended for production but not required. Use `ExecutorType.SUBPROCESS` for development:

```python
agent = Agent(tools=[my_tool], executor=ExecutorType.SUBPROCESS)
```

### Which model should I use?

- **haiku**: Fast, cost-effective for simple operations
- **sonnet**: Balanced performance, good default choice
- **opus**: Maximum capability for complex reasoning

### How do I handle errors?

The framework provides specific exception types:

```python
from claude_agent_toolkit import ConfigurationError, ConnectionError, ExecutionError

try:
    result = await agent.run("task")
except ConfigurationError:
    # Missing OAuth token, invalid config
except ConnectionError:
    # Docker/network issues
except ExecutionError:
    # Tool failures, timeouts
```

### Can I use multiple tools together?

Yes! Claude Code intelligently orchestrates multiple tools:

```python
agent = Agent(tools=[calc_tool, weather_tool, file_tool])
result = await agent.run(
    "Calculate the average temperature and save results to report.txt"
)
```

## Testing

The framework is validated through comprehensive examples rather than traditional unit tests. Each example demonstrates specific capabilities and serves as both documentation and validation.

### Run Examples

```bash
# Clone the repository
git clone https://github.com/cheolwanpark/claude-agent-toolkit.git
cd claude-agent-toolkit

# Set your OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Run different examples
cd src/examples/calculator && python main.py     # Stateful operations, parallel processing
cd src/examples/weather && python main.py       # External API integration
cd src/examples/subprocess && python main.py    # No Docker required
cd src/examples/filesystem && python main.py    # Permission-based file access
cd src/examples/datatransfer && python main.py  # Type-safe data transfer
cd src/examples/mcp && python main.py           # External MCP server integration
```

### Example Structure

```
src/examples/
├── calculator/     # Mathematical operations with state management
├── weather/        # External API integration (OpenWeatherMap)
├── subprocess/     # Subprocess executor demonstration
├── filesystem/     # FileSystemTool with permissions
├── datatransfer/   # DataTransferTool with Pydantic models
├── mcp/           # External MCP server integration (stdio, HTTP)
└── README.md      # Detailed example documentation
```

### Docker Validation

Examples can run with both executors:

```bash
# Docker executor (default)
python main.py

# Subprocess executor (faster startup)
# Examples automatically use subprocess when Docker unavailable
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and validate with examples
4. Run examples to verify functionality
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/cheolwanpark/claude-agent-toolkit.git
cd claude-agent-toolkit
uv sync --group dev

# Validate your changes by running examples
export CLAUDE_CODE_OAUTH_TOKEN='your-token'
cd src/examples/calculator && python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Created by**: [Cheolwan Park](https://github.com/cheolwanpark) • **Blog**: [Project Background](https://blog.codingvillain.com/post/claude-agent-toolkit)

**Links**: [Homepage](https://github.com/cheolwanpark/claude-agent-toolkit) • [Claude Code](https://claude.ai/code) • [Issues](https://github.com/cheolwanpark/claude-agent-toolkit/issues) • [Model Context Protocol](https://modelcontextprotocol.io/)