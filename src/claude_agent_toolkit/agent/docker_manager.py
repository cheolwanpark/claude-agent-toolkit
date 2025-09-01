#!/usr/bin/env python3
# docker_manager.py - Docker client and image management

import docker
from docker.errors import ImageNotFound

from ..constants import DOCKER_HUB_IMAGE
from ..logging import get_logger
from ..exceptions import ConnectionError

logger = get_logger('agent')


class DockerManager:
    """Manages Docker client connection and image management."""
    
    IMAGE_NAME = DOCKER_HUB_IMAGE
    
    def __init__(self):
        """Initialize Docker client and verify connectivity."""
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
    
    def ensure_image(self):
        """Ensure Docker image is available by pulling from Docker Hub."""
        try:
            self.client.images.get(self.IMAGE_NAME)
            logger.debug("Using existing image: %s", self.IMAGE_NAME)
            return
        except ImageNotFound:
            pass
        
        # Pull from Docker Hub
        try:
            logger.info("Pulling image from Docker Hub: %s", self.IMAGE_NAME)
            self.client.images.pull(self.IMAGE_NAME)
            logger.info("Successfully pulled %s", self.IMAGE_NAME)
        except docker.errors.DockerException as e:
            raise ConnectionError(
                f"Failed to pull Docker image {self.IMAGE_NAME} from Docker Hub.\n"
                f"Please ensure the image exists and you have internet connectivity.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Image pull failed with unexpected error: {e}"
            ) from e