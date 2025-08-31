#!/usr/bin/env python3
# docker_manager.py - Docker client and image management

from pathlib import Path
from typing import Optional

import docker
from docker.errors import ImageNotFound


class DockerManager:
    """Manages Docker client connection and image building."""
    
    IMAGE_NAME = "claude-agent:latest"
    
    def __init__(self):
        """Initialize Docker client and verify connectivity."""
        try:
            self.client = docker.from_env()
            self.client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Docker. Please ensure Docker Desktop is running.\n"
                f"Error: {e}"
            )
    
    def ensure_image(self):
        """Build Docker image if it doesn't exist."""
        try:
            self.client.images.get(self.IMAGE_NAME)
            print(f"[agent] Using existing image: {self.IMAGE_NAME}")
        except ImageNotFound:
            print(f"[agent] Building Docker image {self.IMAGE_NAME}...")
            
            # Get directory where this module is located (should be src/agent/)
            # Go up two levels to get to project root
            project_root = Path(__file__).parent.parent.parent
            dockerfile_path = project_root / "docker" / "Dockerfile"
            entrypoint_path = project_root / "docker" / "entrypoint.py"
            
            # Check if required files exist
            if not dockerfile_path.exists():
                raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")
            if not entrypoint_path.exists():
                raise FileNotFoundError(f"entrypoint.py not found at {entrypoint_path}")
            
            # Build image from the docker directory
            try:
                image, logs = self.client.images.build(
                    path=str(dockerfile_path.parent),
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