#!/usr/bin/env python3
# docker_manager.py - Docker client and image management

import docker
from docker.errors import ImageNotFound

from ..constants import DOCKER_HUB_IMAGE


class DockerManager:
    """Manages Docker client connection and image management."""
    
    IMAGE_NAME = DOCKER_HUB_IMAGE
    
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
        """Ensure Docker image is available by pulling from Docker Hub."""
        try:
            self.client.images.get(self.IMAGE_NAME)
            print(f"[agent] Using existing image: {self.IMAGE_NAME}")
            return
        except ImageNotFound:
            pass
        
        # Pull from Docker Hub
        try:
            print(f"[agent] Pulling image from Docker Hub: {self.IMAGE_NAME}")
            self.client.images.pull(self.IMAGE_NAME)
            print(f"[agent] Successfully pulled {self.IMAGE_NAME}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to pull Docker image {self.IMAGE_NAME} from Docker Hub.\n"
                f"Please ensure the image exists and you have internet connectivity.\n"
                f"Error: {e}"
            )