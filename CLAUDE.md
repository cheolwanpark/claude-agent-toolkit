# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Agent Development Kit (claude-adk) - a framework for testing and verifying that Claude Code agents actually call MCP tools correctly. The project provides a Docker-isolated environment where agents can connect to custom MCP tools and verify tool interactions through unpredictable results.

## Architecture

### Core Components
- **Agent Framework** (`agent.py`): Docker-isolated Agent class that runs Claude Code with MCP tool support
- **MCP Tool Framework** (`tool.py`): BaseTool class for creating custom MCP tools with state management
- **Demo Tools** (`main.py`): SecretGeneratorTool and HashComputerTool for verification testing
- **Docker Environment** (`Dockerfile`): Isolated environment with Claude Code CLI and dependencies
- **Entry Point** (`entrypoint.py`): Runs inside Docker containers to execute Claude Code queries

### Tool Development Pattern
1. Create tools by inheriting from `BaseTool`
2. Use `@tool()` decorator to register MCP methods
3. Mark CPU-bound operations with `cpu_bound=True`
4. State is managed through `self.state` dictionary with automatic versioning
5. Tools run as HTTP MCP servers accessible to Docker containers

## Development Commands

### Setup and Running
```bash
# Required: Set OAuth token
export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'

# Install dependencies
uv sync

# Run the main verification demo
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

- Python 3.12+ with `docker`, `fastmcp`, `httpx`, `jsonpatch`, `uvicorn`
- Docker Desktop (must be running)
- Claude Code OAuth token
- Node.js 20 (installed in Docker)

## Development Workflow

1. **Start Docker Desktop** - Required for agent execution
2. **Set OAuth Token** - Export CLAUDE_CODE_OAUTH_TOKEN environment variable  
3. **Create Custom Tools** - Inherit from BaseTool and implement @tool methods
4. **Run Verification** - Use `python main.py` to test agent-tool interactions
5. **Check Tool State** - Verify tools were actually called vs agent hallucinations

## Code Patterns

### Tool Creation
```python
from tool import BaseTool, tool

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
from agent import Agent

# Create and connect to tools
agent = Agent()
agent.connect(my_tool)

# Run agent with prompt
result = await agent.run("Use my_method with parameter 'test'")
```

## Verification Approach

This framework verifies agent behavior by:
1. Using tools that generate unpredictable results (secrets, timestamped hashes)  
2. Checking tool state to confirm methods were actually called
3. Verifying unpredictable values appear in agent responses
4. Testing multi-tool coordination and state management

No traditional unit tests - verification happens through runtime agent testing.