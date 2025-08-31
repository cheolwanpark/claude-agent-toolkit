# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Code Agent Development Kit (claude-adk) - a Python framework for building Claude Code agents with custom tools. The project provides a Docker-isolated environment where Claude Code can orchestrate custom MCP tools for production workflows, leveraging your Claude Code subscription token.

## Architecture

### Core Components
- **Agent Framework** (`src/agent/`): Docker-isolated Agent class that runs Claude Code with MCP tool support
- **MCP Tool Framework** (`src/tool/`): BaseTool class for creating custom MCP tools with state management
- **Example Tools** (`examples/`): Demonstration tools showing practical agent development patterns
- **Docker Environment** (`Dockerfile`, `docker/`): Isolated environment with Claude Code CLI and dependencies

### Agent Development Pattern
1. Create custom tools by inheriting from `BaseTool` 
2. Use `@tool()` decorator to register MCP methods that extend Claude Code's capabilities
3. Mark CPU-bound operations with `cpu_bound=True` for parallel processing
4. State is managed through `self.state` dictionary with automatic versioning and conflict resolution
5. Tools run as HTTP MCP servers that Claude Code can orchestrate in Docker containers
6. Claude Code makes intelligent decisions about which tools to use and when

## Development Commands

### Setup and Running
```bash
# Required: Set OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Install dependencies
uv sync

# Run the demonstration examples
python main.py
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
4. **Build Your Agent** - Use `python main.py` to see examples or create custom agent scripts
5. **Deploy to Production** - Use your Claude Code subscription to run agents at scale

## Code Patterns

### Tool Creation
```python
from src.tool import BaseTool, tool

class MyTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.state = {"my_data": []}
    
    @tool(description="My tool function", cpu_bound=False)
    async def my_method(self, input_param: str) -> dict:
        # Tool logic here
        return {"result": "success"}
```

### Agent Usage
```python
from src.agent import Agent

# Create and connect to tools
agent = Agent()
agent.connect(my_tool)

# Run agent with prompt
result = await agent.run("Use my_method with parameter 'test'")
```

## Claude Code Agent Approach

This framework enables building production Claude Code agents by:
1. Leveraging Claude Code's advanced reasoning with your subscription token
2. Providing custom tools that extend Claude Code's capabilities
3. Managing stateful workflows where Claude Code builds context across tool interactions
4. Orchestrating multi-tool coordination through Claude Code's intelligent decision-making

The framework focuses on production-ready agent development rather than testing - you build agents that use Claude Code's intelligence with your custom tool implementations.