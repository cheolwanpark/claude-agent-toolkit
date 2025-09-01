#!/usr/bin/env python3
# logging.py - Logging module for claude-agent-toolkit library

import logging
import sys
from enum import Enum
from typing import Optional, TextIO, Union


class LogLevel(str, Enum):
    """Logging level options for claude-agent-toolkit."""
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


# Library root logger with NullHandler (Python library best practice)
_root_logger = logging.getLogger('claude_agent_toolkit')
_root_logger.addHandler(logging.NullHandler())
_root_logger.setLevel(logging.DEBUG)

# Global handler reference (created on first use)
_handler: Optional[logging.Handler] = None


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a component.
    
    Args:
        name: Component name (e.g., 'agent', 'tool')
    
    Returns:
        Logger instance with proper hierarchy
    """
    # Simple flat naming: claude_agent_toolkit.agent, claude_agent_toolkit.tool
    return logging.getLogger(f'claude_agent_toolkit.{name}')


def set_logging(
    level: Union[LogLevel, str] = LogLevel.WARNING,
    format: Optional[str] = None,
    stream: TextIO = sys.stderr,
    show_time: bool = False,
    show_level: bool = False
) -> None:
    """Configure logging output for claude-agent-toolkit.
    
    By default, the library uses WARNING level and stderr output with format:
    '[claude_agent_toolkit.component] message'
    
    Args:
        level: Log level (LogLevel enum or string)
        format: Custom format string (overrides other options)
        stream: Output stream (sys.stdout or sys.stderr)
        show_time: Add timestamp to output
        show_level: Add log level to output
    
    Examples:
        # Default: [component] message to stderr, WARNING level
        set_logging()
        
        # Using enum (recommended for IDE support)
        set_logging(LogLevel.INFO)
        set_logging(LogLevel.DEBUG, show_time=True)
        
        # Using string (backward compatible)
        set_logging('INFO')
        
        # Debug with timestamps and levels
        set_logging(LogLevel.DEBUG, show_time=True, show_level=True)
        
        # Custom format
        set_logging(format='%(levelname)s: [%(name)s] %(message)s')
        
        # Output to stdout instead of stderr
        set_logging(LogLevel.INFO, stream=sys.stdout)
    """
    global _handler
    
    # Build format string if not provided
    if format is None:
        parts = []
        if show_time:
            parts.append('%(asctime)s')
        if show_level:
            parts.append('%(levelname)-8s')
        parts.append('[%(name)s]')
        parts.append('%(message)s')
        format = ' '.join(parts)
    
    # Convert LogLevel enum to string if needed
    if isinstance(level, LogLevel):
        level = level.value
    
    # Remove existing handler if present
    if _handler:
        _root_logger.removeHandler(_handler)
    
    # Create new handler with specified configuration
    _handler = logging.StreamHandler(stream)
    _handler.setLevel(getattr(logging, level.upper()))
    _handler.setFormatter(logging.Formatter(format))
    
    # Add handler to root logger
    _root_logger.addHandler(_handler)