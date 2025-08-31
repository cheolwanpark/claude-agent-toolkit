FROM python:3.11-slim

# Platform-aware build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (required for Claude Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Install Python packages
RUN pip install --no-cache-dir \
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
LABEL org.opencontainers.image.source="https://github.com/cheolwanpark/claude-adk"
LABEL org.opencontainers.image.description="Claude Agent Development Kit runtime environment"
LABEL org.opencontainers.image.platform="${TARGETPLATFORM}"

# Default command
CMD ["python", "/app/entrypoint.py"]