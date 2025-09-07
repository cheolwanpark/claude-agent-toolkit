# Subprocess Executor Example

This example demonstrates how to use Claude Agent Toolkit with the **SubprocessExecutor** instead of the default Docker executor.

## Overview

The subprocess executor allows you to run Claude Code agents without Docker by using the `claude-code-sdk` directly in a subprocess. This is useful for:

- Testing tool functionality without Docker overhead
- Environments where Docker is not available
- Development and debugging scenarios
- Lightweight execution with minimal dependencies

## Files

- **`tool.py`**: SimpleTool class with basic echo, history, and status methods
- **`main.py`**: Demo script showing subprocess executor usage
- **`README.md`**: This documentation

## Setup

1. **Set your OAuth token**:
   ```bash
   export CLAUDE_CODE_OAUTH_TOKEN='your-token-here'
   ```
   Get your token from: https://claude.ai/code

2. **Install dependencies** (from project root):
   ```bash
   uv sync
   ```

## Usage

### Run the Demo

```bash
cd src/examples/subprocess
uv run python main.py
```

This will run three demos:
1. Basic echo test
2. Multiple tool interactions 
3. History and status check

### Interactive Mode

```bash
cd src/examples/subprocess
uv run python main.py --interactive
```

Interactive mode lets you chat with the agent and test different commands.

### Example Commands for Interactive Mode

- `"Echo the message 'Hello World'"`
- `"Check the tool status"`
- `"Show me the echo history"`
- `"Echo 'first' then echo 'second' then show status"`

## Key Features Demonstrated

### SubprocessExecutor Usage

The key difference from Docker execution is specifying the executor:

```python
from claude_agent_toolkit import Agent, ExecutorType

# Use subprocess instead of Docker - specify in constructor
agent = Agent(
    tools=[simple_tool],
    executor=ExecutorType.SUBPROCESS  # This is the key line!
)

response = await agent.run(
    "Your prompt here",
    verbose=True
)
```

### Simple Tool Implementation

```python
from claude_agent_toolkit import BaseTool, tool

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.messages = []
    
    @tool(description="Echo a message back with timestamp")
    async def echo(self, message: str) -> Dict[str, Any]:
        # Tool implementation
        pass
```

## Benefits of Subprocess Executor

1. **No Docker Dependency**: Runs without Docker Desktop
2. **Faster Startup**: No container creation overhead
3. **Direct Integration**: Uses claude-code-sdk directly
4. **Easier Debugging**: Simpler execution path for troubleshooting
5. **Lightweight**: Minimal resource usage

## Troubleshooting

### Common Issues

1. **OAuth Token Missing**:
   ```
   ConfigurationError: OAuth token required
   ```
   Solution: Set `CLAUDE_CODE_OAUTH_TOKEN` environment variable

2. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'claude_agent_toolkit'
   ```
   Solution: Run from project root or ensure package is installed

3. **Tool Server Issues**:
   ```
   ConnectionError: Tool server failed to start
   ```
   Solution: Check that ports are available, tools will auto-select ports

### Debugging Tips

1. Enable verbose mode: `verbose=True`
2. Check tool call counts after execution
3. Use interactive mode to test individual commands
4. Compare with Docker executor behavior if needed

## Comparison with Docker Executor

| Feature | Docker Executor | Subprocess Executor |
|---------|----------------|-------------------|
| Dependencies | Docker Desktop | claude-code-sdk only |
| Startup Time | ~2-3 seconds | ~0.5 seconds |
| Resource Usage | Higher (container) | Lower (subprocess) |
| Isolation | Full container isolation | Process isolation only |
| Use Case | Production, full isolation | Development, testing |
| Claude Code SDK | Containerized version | Direct host version |

## Next Steps

After testing this example:

1. Try creating your own custom tools
2. Compare performance with Docker executor
3. Test with more complex tool interactions
4. Explore parallel tool operations with subprocess executor

For production use cases, consider using the Docker executor for better isolation and consistency.