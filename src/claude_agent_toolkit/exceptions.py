"""Exception classes for claude-agent-toolkit."""


class ClaudeAgentError(Exception):
    """Base exception for all claude-agent-toolkit errors."""
    pass


class ConfigurationError(ClaudeAgentError):
    """Raised when configuration is missing or invalid."""
    pass


class ConnectionError(ClaudeAgentError):
    """Raised when connection to services fails."""
    pass


class ExecutionError(ClaudeAgentError):
    """Raised when agent or tool execution fails."""
    pass


