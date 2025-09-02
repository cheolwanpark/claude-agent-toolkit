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

# Model ID mappings (short aliases to full model IDs)
MODEL_ID_MAPPING = {
    "opus": "claude-opus-4-1-20250805",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022"
}


async def main():
    """Main function that runs inside the Docker container."""
    
    # Get configuration from environment variables
    prompt = os.environ.get('AGENT_PROMPT', '')
    tools_json = os.environ.get('MCP_TOOLS', '{}')
    oauth_token = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
    system_prompt = os.environ.get('AGENT_SYSTEM_PROMPT')
    verbose = os.environ.get('AGENT_VERBOSE', '0') == '1'
    model = os.environ.get('ANTHROPIC_MODEL', 'sonnet')
    
    # Apply model ID mapping if needed
    if model in MODEL_ID_MAPPING:
        model = MODEL_ID_MAPPING[model]
    
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
    except json.JSONDecodeError as e:
        print(f"[entrypoint] Warning: Invalid JSON in MCP_TOOLS: {e}", flush=True)
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
            except httpx.TimeoutException:
                print(f"[entrypoint] Health check timeout for {tool_name}", flush=True)
            except httpx.RequestError as e:
                print(f"[entrypoint] Health check connection error for {tool_name}: {e}", flush=True)
            except Exception as e:
                print(f"[entrypoint] Health check failed for {tool_name}: {e}", flush=True)
    
    # Setup Claude Code options with proper MCP configuration
    print(f"[entrypoint] MCP servers config: {json.dumps(mcp_servers, indent=2)}", flush=True)
    print(f"[entrypoint] Tool URLs: {json.dumps(tool_urls, indent=2)}", flush=True)
    print(f"[entrypoint] Using model: {model}", flush=True)
    
    options = ClaudeCodeOptions(
        permission_mode="bypassPermissions",
        mcp_servers=mcp_servers if mcp_servers else {},
        system_prompt=system_prompt,
        model=model
    )
    
    print(f"[entrypoint] Claude Code options - allowed_tools: {options.allowed_tools}", flush=True)
    print(f"[entrypoint] Claude Code options - mcp_servers: {len(options.mcp_servers)} servers", flush=True)
    
    # Initialize collectors
    assistant_responses = []  # For final response text
    result_metadata = None
    error = None
    final_result = None
    
    try:
        print(f"[entrypoint] Starting Claude Code query with {len(mcp_servers)} MCP servers...", flush=True)
        
        message_count = 0
        async for message in query(prompt=prompt, options=options):
            message_count += 1
            if verbose:
                print(f"[entrypoint] Received message #{message_count}: {type(message).__name__}", flush=True)
            
            # Handle UserMessage (may contain ToolResultBlock)
            if isinstance(message, UserMessage):
                if verbose:
                    print(f"[User Message]", flush=True)
                    if isinstance(message.content, str):
                        print(f"  Content: {message.content[:200]}...", flush=True)
                    else:
                        for block in message.content:
                            if isinstance(block, ToolResultBlock):
                                content_preview = block.content[:100] if block.content else 'None'
                                print(f"  ToolResult: {block.tool_use_id} -> {content_preview}...", flush=True)
            
            # Handle AssistantMessage with ALL block types
            elif isinstance(message, AssistantMessage):
                if verbose:
                    print(f"[Assistant Message] Model: {message.model}", flush=True)
                
                for block in message.content:
                    if isinstance(block, TextBlock):
                        assistant_responses.append(block.text)
                        if verbose:
                            print(f"  TextBlock: {block.text[:200]}...", flush=True)
                    
                    elif isinstance(block, ThinkingBlock):
                        if verbose:
                            print(f"  ThinkingBlock: {block.thinking[:200]}...", flush=True)
                    
                    elif isinstance(block, ToolUseBlock):
                        if verbose:
                            print(f"  ToolUse: {block.name}({block.id}) with {list(block.input.keys())}", flush=True)
                    
                    elif isinstance(block, ToolResultBlock):
                        if verbose:
                            status = "ERROR" if block.is_error else "OK"
                            print(f"  ToolResult[{status}]: {block.tool_use_id}", flush=True)
            
            # Handle ResultMessage for metadata
            elif isinstance(message, ResultMessage):
                result_metadata = {
                    "duration_ms": message.duration_ms,
                    "total_cost_usd": message.total_cost_usd,
                    "usage": message.usage,
                    "is_error": message.is_error,
                    "num_turns": message.num_turns
                }
                # Use ResultMessage.result if available
                if message.result:
                    final_result = message.result
                
                if verbose:
                    cost = message.total_cost_usd or 0
                    print(f"[Result Message] Duration: {message.duration_ms}ms, Cost: ${cost:.4f}", flush=True)
            
            # Handle SystemMessage
            elif isinstance(message, SystemMessage):
                if verbose:
                    print(f"[System Message] Type: {message.subtype}", flush=True)
            
    except Exception as e:
        error = str(e)
        print(f"[entrypoint] Error during execution: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # Prepare final output
    if final_result:
        response_text = final_result
    elif assistant_responses:
        response_text = "\n".join(assistant_responses)
    else:
        response_text = "No response generated"
    
    output = {
        "success": error is None and (final_result or len(assistant_responses) > 0),
        "response": response_text,
        "metadata": result_metadata,
        "error": error
    }
    
    # Output final JSON result (this is what the agent will parse)
    print(json.dumps(output))


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())