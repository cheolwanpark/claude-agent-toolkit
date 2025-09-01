#!/usr/bin/env python3
# docker_manager.py - Docker client and image management

import docker
from docker.errors import ImageNotFound

from ..constants import get_versioned_docker_image
from ..logging import get_logger
from ..exceptions import ConnectionError

logger = get_logger('agent')


class DockerManager:
    """Manages Docker client connection and image management."""
    
    def __init__(self):
        """
        Initialize Docker client and verify connectivity.
        
        Note:
            Docker image version automatically matches the installed package version (__version__) for safety.
            No fallback is available - version must match exactly.
        """
        self.IMAGE_NAME = get_versioned_docker_image()
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
                f"Please ensure the exact version image exists and you have internet connectivity.\n"
                f"No fallback is available - version must match exactly for safety.\n"
                f"Error: {e}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Image pull failed with unexpected error: {e}"
            ) from e