#!/usr/bin/env python3
# secret_tool.py - Secret generator tool for verification testing

import secrets
from datetime import datetime
from typing import Dict, Any

from tool import BaseTool, tool


class SecretGeneratorTool(BaseTool):
    """A tool that generates and stores secret values."""
    
    def __init__(self):
        super().__init__()
        self.state = {
            "current_secret": None,
            "secret_history": [],
            "generation_count": 0
        }
    
    @tool(
        description="Generate a new random secret token",
        cpu_bound=False
    )
    async def generate_secret(self) -> Dict[str, Any]:
        """Generate a new secret token."""
        # Generate unpredictable secret
        new_secret = secrets.token_hex(16)
        timestamp = datetime.now().isoformat()
        
        # Store in state
        self.state["current_secret"] = new_secret
        self.state["generation_count"] += 1
        self.state["secret_history"].append({
            "secret": new_secret,
            "timestamp": timestamp,
            "index": self.state["generation_count"]
        })
        
        print(f"\n🔐 [Tool] Generated secret #{self.state['generation_count']}: {new_secret}\n")
        
        return {
            "secret": new_secret,
            "timestamp": timestamp,
            "message": f"Secret #{self.state['generation_count']} generated successfully"
        }
    
    @tool(
        description="Get the current secret token",
        cpu_bound=False
    )
    async def get_current_secret(self) -> Dict[str, Any]:
        """Retrieve the current secret."""
        if self.state["current_secret"] is None:
            return {
                "secret": None,
                "message": "No secret has been generated yet"
            }
        
        return {
            "secret": self.state["current_secret"],
            "generation_count": self.state["generation_count"]
        }
    
    @tool(
        description="Verify if a given token matches the current secret",
        cpu_bound=False
    )
    async def verify_secret(self, token: str) -> Dict[str, Any]:
        """Verify if a token matches the current secret."""
        is_valid = token == self.state["current_secret"]
        
        return {
            "valid": is_valid,
            "provided_token": token,
            "message": "Token is valid!" if is_valid else "Token does not match"
        }