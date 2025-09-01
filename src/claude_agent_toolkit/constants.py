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