#!/usr/bin/env python3
# executor.py - Container execution and result parsing

import json
import uuid
from typing import Any, Dict, Optional

import docker

from ..logging import get_logger
from ..exceptions import ExecutionError
from ..constants import DOCKER_HOST_GATEWAY, CONTAINER_NAME_PREFIX, CONTAINER_UUID_LENGTH

logger = get_logger('agent')


class ContainerExecutor:
    """Handles Docker container execution and result parsing."""
    
    def __init__(self, docker_client: docker.DockerClient, image_name: str):
        """
        Initialize container executor.
        
        Args:
            docker_client: Docker client instance
            image_name: Docker image to use for execution
        """
        self.docker_client = docker_client
        self.image_name = image_name
    
    def execute(self, prompt: str, oauth_token: str, tool_urls: Dict[str, str], system_prompt: Optional[str] = None, verbose: bool = False, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute prompt in Docker container with connected tools.
        
        Args:
            prompt: The instruction for Claude
            oauth_token: Claude Code OAuth token
            tool_urls: Dictionary of tool_name -> url mappings
            system_prompt: Optional system prompt to customize agent behavior
            verbose: If True, enable verbose output in container
            model: Optional model to use for this execution
            
        Returns:
            Dict with success status, response, and metadata
        """
        logger.info("Running with prompt: %s...", prompt[:100])
        
        # Prepare environment variables
        environment = {
            'CLAUDE_CODE_OAUTH_TOKEN': oauth_token,
            'AGENT_PROMPT': prompt,
            'AGENT_VERBOSE': '1' if verbose else '0'
        }
        
        # Add system prompt if provided
        if system_prompt:
            environment['AGENT_SYSTEM_PROMPT'] = system_prompt
        
        # Add model if provided
        if model:
            environment['ANTHROPIC_MODEL'] = model
        
        # Add all connected tools as separate environment variables
        if tool_urls:
            # Pass tools as JSON for easier parsing in entrypoint
            environment['MCP_TOOLS'] = json.dumps(tool_urls)
            logger.info("Connected tools: %s", list(tool_urls.keys()))
        
        try:
            # Run container with entrypoint.py
            container_name = f"{CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:CONTAINER_UUID_LENGTH]}"
            
            logger.debug("Starting container %s", container_name)
            
            result = self.docker_client.containers.run(
                image=self.image_name,
                name=container_name,
                command="python /app/entrypoint.py",  # Use the built-in entrypoint
                environment=environment,
                extra_hosts={'host.docker.internal': DOCKER_HOST_GATEWAY},
                auto_remove=not verbose,  # Automatically remove container if not verbose mode
                stdout=True,
                stderr=True,
                detach=False
            )
            
            # Parse output
            output = result.decode('utf-8').strip()
            
            # Find JSON in output (it might have other logs)
            return self._parse_container_output(output)
                
        except docker.errors.DockerException as e:
            logger.error("Docker execution failed: %s", e)
            raise ExecutionError(f"Agent execution failed due to Docker error: {e}") from e
        except Exception as e:
            logger.error("Execution failed: %s", e)
            raise ExecutionError(f"Agent execution failed: {e}") from e
    
    def _parse_container_output(self, output: str) -> Dict[str, Any]:
        """
        Parse container output to extract JSON result.
        
        Args:
            output: Raw container output
            
        Returns:
            Parsed result dictionary
        """
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
            logger.info("Execution completed successfully")
            return json_output
        else:
            logger.warning("No valid JSON output found")
            raise ExecutionError(f"Failed to parse agent output. Raw output: {output or 'No output'}")