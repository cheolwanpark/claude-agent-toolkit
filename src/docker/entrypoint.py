#!/usr/bin/env python3
# entrypoint.py - Entry point script that runs inside the Docker container

import asyncio
import json
import os
from claude_code_sdk import (
    query, ClaudeCodeOptions, 
    AssistantMessage, UserMessage, SystemMessage, ResultMessage,
    TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock
)

# Model ID mappings removed - now handled in executor.py


async def main():
    """Main function that runs inside the Docker container."""
    
    # Get configuration from environment variables
    prompt = os.environ.get('AGENT_PROMPT', '')
    tools_json = os.environ.get('MCP_TOOLS', '{}')
    oauth_token = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
    system_prompt = os.environ.get('AGENT_SYSTEM_PROMPT')
    verbose = os.environ.get('AGENT_VERBOSE', '0') == '1'
    model = os.environ.get('ANTHROPIC_MODEL', None)
    
    if not prompt:
        print("ERROR: No prompt provided - AGENT_PROMPT environment variable is empty", file=sys.stderr, flush=True)
        return
    
    if not oauth_token:
        print("ERROR: No OAuth token provided - CLAUDE_CODE_OAUTH_TOKEN environment variable is empty", file=sys.stderr, flush=True)
        return
    
    # Parse tools configuration
    try:
        tool_urls = json.loads(tools_json)
    except json.JSONDecodeError as e:
        print(f"[entrypoint] Warning: Invalid JSON in MCP_TOOLS: {e}", file=sys.stderr, flush=True)
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
            print(f"[entrypoint] Configured HTTP MCP server {tool_name} at {tool_url}", file=sys.stderr, flush=True)
            
            # Test connectivity to MCP server
            try:
                import httpx
                with httpx.Client(timeout=5.0) as client:
                    health_url = tool_url.replace('/mcp', '/health')
                    response = client.get(health_url)
                    print(f"[entrypoint] Health check for {tool_name}: {response.status_code}", file=sys.stderr, flush=True)
            except httpx.TimeoutException:
                print(f"[entrypoint] Health check timeout for {tool_name}", file=sys.stderr, flush=True)
            except httpx.RequestError as e:
                print(f"[entrypoint] Health check connection error for {tool_name}: {e}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[entrypoint] Health check failed for {tool_name}: {e}", file=sys.stderr, flush=True)
    
    # Setup Claude Code options with proper MCP configuration
    print(f"[entrypoint] MCP servers config: {json.dumps(mcp_servers, indent=2)}", file=sys.stderr, flush=True)
    print(f"[entrypoint] Tool URLs: {json.dumps(tool_urls, indent=2)}", file=sys.stderr, flush=True)
    print(f"[entrypoint] Using model: {model}", file=sys.stderr, flush=True)
    
    options = ClaudeCodeOptions(
        permission_mode="bypassPermissions",
        mcp_servers=mcp_servers if mcp_servers else {},
        system_prompt=system_prompt,
        model=model
    )
    
    print(f"[entrypoint] Claude Code options - allowed_tools: {options.allowed_tools}", file=sys.stderr, flush=True)
    print(f"[entrypoint] Claude Code options - mcp_servers: {len(options.mcp_servers)} servers", file=sys.stderr, flush=True)
    
    def serialize_message(message):
        """Convert a claude-code-sdk message to a serializable dict."""
        message_dict = {
            "type": type(message).__name__
        }
        
        # Add common fields
        if hasattr(message, 'content'):
            if isinstance(message.content, str):
                message_dict["content"] = message.content
            else:
                # Handle list of blocks
                content_list = []
                for block in message.content:
                    block_dict = {"type": type(block).__name__}
                    
                    if hasattr(block, 'text'):
                        block_dict["text"] = block.text
                    if hasattr(block, 'thinking'):
                        block_dict["thinking"] = block.thinking
                    if hasattr(block, 'name'):
                        block_dict["name"] = block.name
                    if hasattr(block, 'id'):
                        block_dict["id"] = block.id
                    if hasattr(block, 'input'):
                        block_dict["input"] = block.input
                    if hasattr(block, 'tool_use_id'):
                        block_dict["tool_use_id"] = block.tool_use_id
                    if hasattr(block, 'content'):
                        block_dict["content"] = block.content
                    if hasattr(block, 'is_error'):
                        block_dict["is_error"] = block.is_error
                        
                    content_list.append(block_dict)
                message_dict["content"] = content_list
        
        if hasattr(message, 'model'):
            message_dict["model"] = message.model
        if hasattr(message, 'subtype'):
            message_dict["subtype"] = message.subtype
        if hasattr(message, 'result'):
            message_dict["result"] = message.result
        if hasattr(message, 'duration_ms'):
            message_dict["duration_ms"] = message.duration_ms
        if hasattr(message, 'total_cost_usd'):
            message_dict["total_cost_usd"] = message.total_cost_usd
        if hasattr(message, 'usage'):
            message_dict["usage"] = message.usage
        if hasattr(message, 'is_error'):
            message_dict["is_error"] = message.is_error
        if hasattr(message, 'num_turns'):
            message_dict["num_turns"] = message.num_turns
            
        return message_dict
    
    try:
        print(f"[entrypoint] Starting Claude Code query with {len(mcp_servers)} MCP servers...", file=sys.stderr, flush=True)
        
        async for message in query(prompt=prompt, options=options):
            # Serialize and output each message as JSON to stdout
            message_dict = serialize_message(message)
            print(json.dumps(message_dict), flush=True)
            
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())