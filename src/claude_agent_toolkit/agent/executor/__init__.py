#!/usr/bin/env python3
# __init__.py - Executor module exports

from enum import Enum

from .base import BaseExecutor
from .docker import DockerExecutor
from .subprocess import SubprocessExecutor

# Maintain backward compatibility
ContainerExecutor = DockerExecutor


class ExecutorType(Enum):
    """Enumeration of available executor types."""
    DOCKER = "docker"
    SUBPROCESS = "subprocess"


def create_executor(executor_type: ExecutorType = ExecutorType.DOCKER) -> BaseExecutor:
    """
    Factory function to create executor instances.
    
    Args:
        executor_type: Type of executor to create (defaults to DOCKER)
        
    Returns:
        BaseExecutor instance
        
    Raises:
        ValueError: If unknown executor type is provided
    """
    if executor_type == ExecutorType.DOCKER:
        return DockerExecutor()
    elif executor_type == ExecutorType.SUBPROCESS:
        return SubprocessExecutor()
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


__all__ = [
    'BaseExecutor', 
    'DockerExecutor', 
    'SubprocessExecutor', 
    'ContainerExecutor',
    'ExecutorType',
    'create_executor'
]