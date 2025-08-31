#!/usr/bin/env python3
# entrypoint.py - Entry point script that runs inside the Docker container

import asyncio
import json
import os
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock


async def main():
    """Main function that runs inside the Docker container."""
    
    # Get configuration from environment variables
    prompt = os.environ.get('AGENT_PROMPT', '')
    tools_json = os.environ.get('MCP_TOOLS', '{}')
    oauth_token = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
    system_prompt = os.environ.get('AGENT_SYSTEM_PROMPT')
    
    if not prompt:
        print(json.dumps({
            "success": False,
            "response": "No prompt provided",
            "error": "AGENT_PROMPT environment variable is empty"
        }))
        return
    
    if not oauth_token:
        print(json.dumps({
            "success": False,
            "response": "No OAuth token provided",
            "error": "CLAUDE_CODE_OAUTH_TOKEN environment variable is empty"
        }))
        return
    
    # Parse tools configuration
    try:
        tool_urls = json.loads(tools_json)
    except json.JSONDecodeError:
        tool_urls = {}
    
    # Configure MCP servers using HTTP configuration
    mcp_servers = {}
    if tool_urls:
        # Create proper HTTP MCP server configuration for each tool
        for tool_name, tool_url in tool_urls.items():
            # Use the HTTP configuration type
            mcp_servers[tool_name.lower()] = {
                "type": "http",
                "url": tool_url,
                "headers": {}  # Add any necessary headers here
            }
            print(f"[entrypoint] Configured HTTP MCP server {tool_name} at {tool_url}", flush=True)
            
            # Test connectivity to MCP server
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    health_url = tool_url.replace('/mcp', '/health')
                    response = client.get(health_url)
                    print(f"[entrypoint] Health check for {tool_name}: {response.status_code}", flush=True)
            except Exception as e:
                print(f"[entrypoint] Health check failed for {tool_name}: {e}", flush=True)
    
    # Setup Claude Code options with proper MCP configuration
    print(f"[entrypoint] MCP servers config: {json.dumps(mcp_servers, indent=2)}", flush=True)
    print(f"[entrypoint] Tool URLs: {json.dumps(tool_urls, indent=2)}", flush=True)
    
    options = ClaudeCodeOptions(
        permission_mode="bypassPermissions",
        mcp_servers=mcp_servers if mcp_servers else {},
        system_prompt=system_prompt
    )
    
    print(f"[entrypoint] Claude Code options - allowed_tools: {options.allowed_tools}", flush=True)
    print(f"[entrypoint] Claude Code options - mcp_servers: {len(options.mcp_servers)} servers", flush=True)
    
    # Collect results
    results = []
    error = None
    
    try:
        print(f"[entrypoint] Starting Claude Code query with {len(mcp_servers)} MCP servers...", flush=True)
        
        message_count = 0
        async for message in query(prompt=prompt, options=options):
            message_count += 1
            print(f"[entrypoint] Received message #{message_count}: {type(message).__name__}", flush=True)
            
            if isinstance(message, AssistantMessage):
                print(f"[entrypoint] AssistantMessage content blocks: {len(message.content)}", flush=True)
                for i, block in enumerate(message.content):
                    print(f"[entrypoint] Block {i}: {type(block).__name__}", flush=True)
                    if isinstance(block, TextBlock):
                        results.append(block.text)
                        print(f"[entrypoint] Text block content (first 200 chars): {block.text[:200]}", flush=True)
            else:
                print(f"[entrypoint] Non-assistant message: {message}", flush=True)
            
    except Exception as e:
        error = str(e)
        print(f"[entrypoint] Error during execution: {e}", flush=True)
        import traceback
        traceback.print_exc()
        results.append(f"Error: {e}")
    
    # Prepare final output
    if results:
        response_text = "\n".join(results)
    else:
        response_text = "No response generated"
    
    output = {
        "success": error is None and len(results) > 0,
        "response": response_text,
        "error": error
    }
    
    # Output final JSON result (this is what the agent will parse)
    print(json.dumps(output))


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())