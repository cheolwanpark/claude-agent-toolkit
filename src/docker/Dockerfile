FROM python:3.11-slim

# Platform-aware build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG VERSION

# Install system dependencies and Node.js in single layer
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install Python packages with optimized compilation
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --only-binary=:all: --prefer-binary \
    claude-code-sdk \
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