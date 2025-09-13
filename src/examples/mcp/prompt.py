#!/usr/bin/env python3
# mcp/prompt.py - Prompts and test scenarios for MCP Everything server demo

# System prompt for the agent explaining Everything server capabilities
MCP_SYSTEM_PROMPT = """You are a helpful assistant with access to the Everything MCP Server tools.

The Everything MCP Server provides these basic tools:
- echo: Echo back input messages
- add: Add two numbers together

Always use the available MCP tools when appropriate. Be clear and helpful in your responses.
"""

# Welcome message for the demo
WELCOME_MESSAGE = """
🌟 Everything MCP Server Demo - Transport Connectivity
=====================================================

This demo showcases MCP transport connectivity using basic tools:
1. Stdio Transport - Direct process communication
2. HTTP Transport - HTTP connections

Each transport will demonstrate:
- Connection establishment and tool discovery
- Basic echo functionality to verify connectivity
- Simple math operations to test tool execution
"""

# Test scenarios for transport connectivity
TEST_SCENARIOS = {
    "connectivity": {
        "name": "Connectivity Test",
        "prompt": "Use the echo tool to say 'Connected successfully!' to verify the transport is working."
    },

    "basic_math": {
        "name": "Basic Math Operation",
        "prompt": "Use the add tool to calculate 15 + 27 to test tool execution."
    }
}

def get_demo_prompt(scenario: str) -> str:
    """Get a formatted prompt for a specific test scenario."""
    if scenario not in TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(TEST_SCENARIOS.keys())}")

    return TEST_SCENARIOS[scenario]["prompt"]

def get_scenario_name(scenario: str) -> str:
    """Get the display name for a test scenario."""
    if scenario not in TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(TEST_SCENARIOS.keys())}")

    return TEST_SCENARIOS[scenario]["name"]

def format_demo_output(transport: str, scenario: str, response: str, truncate: int = 600) -> str:
    """Format demo output for consistent display."""
    scenario_name = get_scenario_name(scenario)
    truncated = response[:truncate] + ("..." if len(response) > truncate else "")

    return f"""
📋 {transport.upper()} Transport - {scenario_name}
{'-' * (len(transport) + len(scenario_name) + 15)}
Response: {truncated}
"""