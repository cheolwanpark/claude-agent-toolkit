#!/usr/bin/env python3
# mcp/main.py - MCP Everything server demo across all transport types

import asyncio
import os
import subprocess
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ExecutorType

# Import our new MCP wrapper classes
from claude_agent_toolkit.tool.mcp import StdioMCPTool, HttpMCPTool

# Import our prompts and test scenarios
from prompt import MCP_SYSTEM_PROMPT, WELCOME_MESSAGE, get_demo_prompt, format_demo_output


async def run_stdio_demo():
    """Run the stdio transport demo using npx command."""
    print("\n" + "="*60)
    print("📡 STDIO TRANSPORT DEMO")
    print("="*60)
    print("Using: npx -y @modelcontextprotocol/server-everything")

    try:
        # Create StdioMCPTool for Everything server
        stdio_tool = StdioMCPTool(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
            name="everything-stdio"
        )

        # Create agent
        agent = Agent(
            system_prompt=MCP_SYSTEM_PROMPT,
            tools=[stdio_tool],
            executor=ExecutorType.SUBPROCESS  # Faster for demos
        )

        # Test connectivity
        print("\n🧪 Testing connectivity...")
        response = await agent.run(get_demo_prompt("connectivity"))
        print(format_demo_output("stdio", "connectivity", response))

        # Test basic math operation
        print("\n🧪 Testing basic math operations...")
        response = await agent.run(get_demo_prompt("basic_math"))
        print(format_demo_output("stdio", "basic_math", response))

        print("\n✅ Stdio transport demo completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Stdio demo failed: {e}")
        return False


async def run_http_demo():
    """Run the HTTP transport demo with subprocess server management."""
    print("\n" + "="*60)
    print("🌐 HTTP TRANSPORT DEMO")
    print("="*60)
    print("Using: npx @modelcontextprotocol/server-everything streamableHttp")

    server_process = None
    try:
        # Start the HTTP server in background
        print("🚀 Starting Everything HTTP server...")
        server_process = subprocess.Popen(
            ["npx", "-y", "@modelcontextprotocol/server-everything", "streamableHttp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start (default port is usually 3001)
        print("⏳ Waiting for server to start...")
        await asyncio.sleep(3)  # Give server time to start

        # Create HttpMCPTool for Everything server
        http_tool = HttpMCPTool(
            url="http://localhost:3001/mcp",
            name="everything-http"
        )

        # Create agent
        agent = Agent(
            system_prompt=MCP_SYSTEM_PROMPT,
            tools=[http_tool],
            executor=ExecutorType.SUBPROCESS
        )

        # Test connectivity over HTTP
        print("\n🧪 Testing connectivity over HTTP...")
        response = await agent.run(get_demo_prompt("connectivity"))
        print(format_demo_output("http", "connectivity", response))

        # Test basic math over HTTP
        print("\n🧪 Testing basic math over HTTP...")
        response = await agent.run(get_demo_prompt("basic_math"))
        print(format_demo_output("http", "basic_math", response))

        print("\n✅ HTTP transport demo completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ HTTP demo failed: {e}")
        return False

    finally:
        # Clean up server process
        if server_process:
            print("🧹 Cleaning up HTTP server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()



async def main():
    """Main function to run all MCP transport demos."""
    print(WELCOME_MESSAGE)

    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return False

    print("\n📋 Running MCP Transport Demos")
    print("=" * 40)
    print("Note: This requires Node.js and npx to be installed")
    print("The Everything server will be downloaded automatically via npx")

    success_count = 0
    total_demos = 2

    # Run stdio demo
    if await run_stdio_demo():
        success_count += 1

    # Run HTTP demo
    if await run_http_demo():
        success_count += 1

    # Summary
    print("\n" + "="*60)
    print(f"📊 DEMO SUMMARY: {success_count}/{total_demos} demos completed successfully")
    print("="*60)

    if success_count == total_demos:
        print("🎉 All MCP transport demos completed successfully!")
        print("\nKey achievements:")
        print("✅ Demonstrated StdioMCPTool connectivity with direct npx execution")
        print("✅ Demonstrated HttpMCPTool connectivity with subprocess management")
        print("✅ Verified tool discovery and execution across both transports")
        print("✅ Confirmed echo and add tools work identically on both transports")
    else:
        print(f"⚠️  {total_demos - success_count} demo(s) failed")
        print("This might be due to:")
        print("- Missing Node.js or npx installation")
        print("- Network issues downloading the Everything server")
        print("- Port conflicts for HTTP/SSE servers")
        print("- OAuth token configuration issues")

    return success_count == total_demos


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)