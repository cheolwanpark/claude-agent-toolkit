#!/usr/bin/env python3
# constants.py - Claude Agent Toolkit constants and configuration

"""
Constants and configuration values for the Claude Agent Toolkit.
"""

# Docker Hub repository configuration
DOCKER_HUB_REPO = "cheolwanpark/claude-agent"
DEFAULT_IMAGE_TAG = "latest"

# Docker image name for Docker Hub
DOCKER_HUB_IMAGE = f"{DOCKER_HUB_REPO}:{DEFAULT_IMAGE_TAG}"

# Local image name (fallback)
LOCAL_IMAGE_NAME = "claude-agent:latest"

# Docker networking configuration
DOCKER_LOCALHOST_MAPPINGS = {
    "localhost": "host.docker.internal",
    "127.0.0.1": "host.docker.internal",
}
DOCKER_HOST_GATEWAY = "host-gateway"

# Environment variable names
ENV_CLAUDE_CODE_OAUTH_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"

# Container naming
CONTAINER_NAME_PREFIX = "agent-"
CONTAINER_UUID_LENGTH = 8