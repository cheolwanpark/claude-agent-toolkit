FROM python:3.11-slim

# Platform-aware build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG VERSION

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (claude-agent-sdk includes Claude Code CLI)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --only-binary=:all: --prefer-binary \
    claude-agent-sdk>=0.1.3 \
    httpx \
    fastmcp \
    anyio

# Create a non-root user
RUN useradd -m -s /bin/bash claudeuser

# Create app directory and set ownership
WORKDIR /app
RUN chown claudeuser:claudeuser /app

# Copy entrypoint script with proper ownership
COPY --chown=claudeuser:claudeuser entrypoint.py /app/entrypoint.py

# Create and set ownership of workspace directory
RUN mkdir -p /workspace && chown claudeuser:claudeuser /workspace
WORKDIR /workspace

# Switch to non-root user
USER claudeuser

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Add metadata
LABEL org.opencontainers.image.source="https://github.com/cheolwanpark/claude-agent-toolkit"
LABEL org.opencontainers.image.description="Claude Agent Toolkit runtime environment"
LABEL org.opencontainers.image.platform="${TARGETPLATFORM}"
LABEL org.opencontainers.image.version="${VERSION}"

# Default command
CMD ["python", "/app/entrypoint.py"]