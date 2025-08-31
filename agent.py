#!/usr/bin/env python3
# agent.py - Framework module for Docker-isolated Claude agents

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

import docker
from docker.errors import ImageNotFound


class Agent:
    """
    Docker-isolated Agent that runs Claude Code with MCP tool support.
    
    Usage:
        agent = Agent(oauth_token="...")
        agent.connect(tool1)
        agent.connect(tool2)
        result = await agent.run("Your prompt")
    """
    
    IMAGE_NAME = "claude-agent:latest"
    
    def __init__(self, oauth_token: Optional[str] = None):
        """
        Initialize the Agent.
        
        Args:
            oauth_token: Claude Code OAuth token (or use CLAUDE_CODE_OAUTH_TOKEN env var)
        """
        # Check Docker availability
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker. Please ensure Docker Desktop is running.\n"
                f"Error: {e}"
            )
        
        self.oauth_token = oauth_token or os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '')
        self.tool_urls: Dict[str, str] = {}  # tool_name -> url mapping
        
        if not self.oauth_token:
            raise ValueError("OAuth token required: pass oauth_token or set CLAUDE_CODE_OAUTH_TOKEN")
        
        # Ensure Docker image exists
        self._ensure_image()
    
    def _ensure_image(self):
        """Build Docker image if it doesn't exist."""
        try:
            self.docker_client.images.get(self.IMAGE_NAME)
            print(f"[agent] Using existing image: {self.IMAGE_NAME}")
        except ImageNotFound:
            print(f"[agent] Building Docker image {self.IMAGE_NAME}...")
            
            # Get directory where agent.py is located
            framework_dir = Path(__file__).parent
            dockerfile_path = framework_dir / "Dockerfile"
            entrypoint_path = framework_dir / "entrypoint.py"
            
            # Check if required files exist
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            if not entrypoint_path.exists():
                raise FileNotFoundError(f"entrypoint.py not found at {entrypoint_path}")
            
            # Build image from the framework directory
            try:
                image, logs = self.docker_client.images.build(
                    path=str(framework_dir),
                    tag=self.IMAGE_NAME,
                    rm=True,
                    forcerm=True
                )
                
                # Print build logs
                for log in logs:
                    if 'stream' in log:
                        print(log['stream'].strip())
                
                print(f"[agent] Successfully built {self.IMAGE_NAME}")
            except Exception as e:
                raise RuntimeError(f"Failed to build Docker image: {e}")
    
    def connect(self, tool: Any) -> 'Agent':
        """
        Connect to an MCP tool server. Can be called multiple times for multiple tools.
        
        Args:
            tool: Tool instance with connection_url property
            
        Returns:
            Self for chaining
        """
        if not hasattr(tool, 'connection_url'):
            raise ValueError("Tool must have 'connection_url' property")
        
        # Get tool name (class name)
        tool_name = tool.__class__.__name__
        
        # Rewrite localhost URLs for Docker container access
        url = tool.connection_url
        url = url.replace('localhost', 'host.docker.internal') 
        url = url.replace('127.0.0.1', 'host.docker.internal')
        
        self.tool_urls[tool_name] = url
        print(f"[agent] Connected to {tool_name} at {url}")
        
        return self
    
    async def run(self, prompt: str) -> Dict[str, Any]:
        """
        Run the agent with the given prompt.
        
        Args:
            prompt: The instruction for Claude
            
        Returns:
            Dict with success status and response
        """
        print(f"[agent] Running with prompt: {prompt[:100]}...")
        
        # Prepare environment variables
        environment = {
            'CLAUDE_CODE_OAUTH_TOKEN': self.oauth_token,
            'AGENT_PROMPT': prompt
        }
        
        # Add all connected tools as separate environment variables
        if self.tool_urls:
            # Pass tools as JSON for easier parsing in entrypoint
            environment['MCP_TOOLS'] = json.dumps(self.tool_urls)
            print(f"[agent] Connected tools: {list(self.tool_urls.keys())}")
        
        try:
            # Run container with entrypoint.py
            container_name = f"agent-{uuid.uuid4().hex[:8]}"
            
            print(f"[agent] Starting container {container_name}")
            
            result = self.docker_client.containers.run(
                image=self.IMAGE_NAME,
                name=container_name,
                command="python /app/entrypoint.py",  # Use the built-in entrypoint
                environment=environment,
                extra_hosts={'host.docker.internal': 'host-gateway'},
                remove=False,  # Keep container for debugging
                stdout=True,
                stderr=True,
                detach=False
            )
            
            # Parse output
            output = result.decode('utf-8').strip()
            
            # Find JSON in output (it might have other logs)
            lines = output.split('\n')
            json_output = None
            
            for line in reversed(lines):  # Check from the end
                if line.strip().startswith('{'):
                    try:
                        json_output = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if json_output:
                print(f"[agent] Execution completed successfully")
                return json_output
            else:
                print(f"[agent] No valid JSON output found")
                return {
                    "success": False,
                    "response": output or "No output from agent",
                    "error": "Failed to parse agent output"
                }
                
        except Exception as e:
            print(f"[agent] Execution failed: {e}")
            return {
                "success": False,
                "response": f"Agent execution failed: {str(e)}",
                "error": str(e)
            }