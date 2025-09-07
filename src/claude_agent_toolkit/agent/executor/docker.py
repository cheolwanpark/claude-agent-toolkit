#!/usr/bin/env python3
# docker.py - Docker executor with integrated Docker management

import json
import uuid
from typing import Dict, List, Optional

import docker
from docker.errors import ImageNotFound

from .base import BaseExecutor
from ...constants import (
    get_versioned_docker_image, DOCKER_HOST_GATEWAY, 
    CONTAINER_NAME_PREFIX, CONTAINER_UUID_LENGTH, MODEL_ID_MAPPING
)
from ...logging import get_logger
from ...exceptions import ConfigurationError, ConnectionError, ExecutionError
from ..response_handler import ResponseHandler

logger = get_logger('agent')


class DockerExecutor(BaseExecutor):
    """Docker-based executor with integrated Docker client and image management."""
    
    def __init__(self):
        """
        Initialize Docker executor with client and image management.
        
        Note:
            Docker image version automatically matches the installed package version (__version__) for safety.
            No fallback is available - version must match exactly.
        """
        self.image_name = get_versioned_docker_image()
        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException as e:
            raise ConnectionError(
                f"Cannot connect to Docker. Please ensure Docker Desktop is running.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Docker connection failed with unexpected error: {e}"
            ) from e
        
        # Ensure Docker image is available
        self.ensure_image()
    
    def ensure_image(self):
        """Ensure Docker image is available by pulling from Docker Hub."""
        try:
            self.client.images.get(self.image_name)
            logger.debug("Using existing image: %s", self.image_name)
            return
        except ImageNotFound:
            pass
        
        # Pull from Docker Hub
        try:
            logger.info("Pulling image from Docker Hub: %s", self.image_name)
            self.client.images.pull(self.image_name)
            logger.info("Successfully pulled %s", self.image_name)
        except docker.errors.DockerException as e:
            raise ConnectionError(
                f"Failed to pull Docker image {self.image_name} from Docker Hub.\n"
                f"Please ensure the exact version image exists and you have internet connectivity.\n"
                f"No fallback is available - version must match exactly for safety.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Image pull failed with unexpected error: {e}"
            ) from e
    
    async def run(
        self, 
        prompt: str, 
        oauth_token: str, 
        tool_urls: Dict[str, str], 
        allowed_tools: Optional[List[str]] = None, 
        system_prompt: Optional[str] = None, 
        verbose: bool = False, 
        model: Optional[str] = None
    ) -> str:
        """
        Execute prompt in Docker container with connected tools.
        
        Args:
            prompt: The instruction for Claude
            oauth_token: Claude Code OAuth token
            tool_urls: Dictionary of tool_name -> url mappings
            allowed_tools: List of allowed tool IDs (mcp__servername__toolname format)
            system_prompt: Optional system prompt to customize agent behavior
            verbose: If True, enable verbose output in container
            model: Optional model to use for this execution
            
        Returns:
            Response string from Claude
            
        Raises:
            ConfigurationError: If OAuth token or configuration is invalid
            ConnectionError: If Docker connection fails
            ExecutionError: If container execution fails
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
        
        # Add model if provided - apply model ID mapping if needed
        if model:
            # Apply model ID mapping if model is a known alias
            mapped_model = MODEL_ID_MAPPING.get(model, model)
            environment['ANTHROPIC_MODEL'] = mapped_model
        
        # Add all connected tools as separate environment variables
        if tool_urls:
            # Pass tools as JSON for easier parsing in entrypoint
            environment['MCP_TOOLS'] = json.dumps(tool_urls)
            logger.info("Connected tools: %s", list(tool_urls.keys()))
        
        # Add allowed tools list
        if allowed_tools:
            environment['ALLOWED_TOOLS'] = json.dumps(allowed_tools)
            logger.info("Allowed tools: %d tools discovered", len(allowed_tools))
        
        if not oauth_token:
            raise ConfigurationError("OAuth token is required")
        
        try:
            # Run container with streaming
            container_name = f"{CONTAINER_NAME_PREFIX}{uuid.uuid4().hex[:CONTAINER_UUID_LENGTH]}"
            
            logger.debug("Starting container %s", container_name)
            
            container = self.client.containers.run(
                image=self.image_name,
                name=container_name,
                command="python /app/entrypoint.py",
                environment=environment,
                extra_hosts={'host.docker.internal': DOCKER_HOST_GATEWAY},
                stdout=True,
                stderr=True,
                stream=True,
                detach=False,
                remove=True
            )
            
            handler = ResponseHandler()
            
            # Process stream
            for chunk in container:
                if chunk:
                    chunk_text = chunk.decode('utf-8')
                    for line in chunk_text.splitlines():
                        result = handler.handle(line, verbose)
                        if result:
                            logger.info("Execution completed successfully")
                            return result  # Return string directly
                        
            # If we get here, no ResultMessage was received
            if handler.text_responses:
                logger.info("Execution completed with text responses")
                return '\n'.join(handler.text_responses)
            else:
                raise ExecutionError("No response received from Claude")
                
        except docker.errors.ContainerError as e:
            stderr = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error("Container failed: %s", stderr)
            raise ExecutionError(f"Container failed: {stderr}") from e
            
        except docker.errors.DockerException as e:
            logger.error("Docker connection failed: %s\n", e)
            raise ConnectionError(f"Docker connection failed: {e}") from e
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, ConnectionError, ExecutionError)):
                raise  # Re-raise our exceptions
            logger.error("Unexpected error: %s", e)
            raise ExecutionError(f"Unexpected error: {e}") from e