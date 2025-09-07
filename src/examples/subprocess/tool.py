#!/usr/bin/env python3
# subprocess/tool.py - Message reflector tool for testing subprocess executor

import time
from typing import Dict, Any

from claude_agent_toolkit import BaseTool, tool


class MessageReflector(BaseTool):
    """A message reflector tool for testing the subprocess executor with distinct naming."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.messages = []
    
    @tool(description="Reflect a message back with timestamp and metadata")
    async def reflect_message(self, message: str) -> Dict[str, Any]:
        """Reflect a message back with additional metadata using the MessageReflector tool."""
        self.call_count += 1
        timestamp = time.time()
        
        response_data = {
            "reflected_message": message,
            "timestamp": timestamp,
            "call_number": self.call_count,
            "message": f"Successfully reflected: '{message}'"
        }
        
        # Store the message for history
        self.messages.append({
            "input": message,
            "timestamp": timestamp,
            "call_number": self.call_count
        })
        
        print(f"\n🔄 [MessageReflector] Reflection #{self.call_count}: '{message}'\n")
        
        return response_data
    
    @tool(description="Get the history of message reflections from the MessageReflector tool")
    async def get_reflection_history(self) -> Dict[str, Any]:
        """Get the history of all message reflections."""
        return {
            "total_reflections": self.call_count,
            "messages": self.messages,
            "status": f"MessageReflector tool has been called {self.call_count} times"
        }
    
    @tool(description="Get the current status of the MessageReflector tool")
    async def get_tool_status(self) -> Dict[str, Any]:
        """Get current status of the MessageReflector tool."""
        return {
            "active": True,
            "reflection_count": self.call_count,
            "last_reflected_message": self.messages[-1]["input"] if self.messages else None,
            "uptime_info": "MessageReflector tool is running successfully"
        }