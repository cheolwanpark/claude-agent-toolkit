#!/usr/bin/env python3
# constants.py - Claude Agent Toolkit constants and configuration

"""
Constants and configuration values for the Claude Agent Toolkit.
"""

# Docker Hub repository configuration
DOCKER_HUB_REPO = "cheolwanpark/claude-agent-toolkit"

def get_versioned_docker_image():
    """
    Get Docker image name using the current package version for safety.
    
    Returns:
        Docker image name with version tag matching the installed package exactly
        
    Note:
        This enforces strict version consistency between PyPI package and Docker image.
        No fallback is available - version must match exactly.
    """
    from . import __version__
    return f"{DOCKER_HUB_REPO}:{__version__}"

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

# Model ID mappings (short aliases to full model IDs)
MODEL_ID_MAPPING = {
    "opus": "claude-opus-4-1-20250805",
    "sonnet": "claude-sonnet-4-20250514",
    "haiku": "claude-3-5-haiku-20241022"
}