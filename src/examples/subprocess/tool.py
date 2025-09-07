#!/usr/bin/env python3
# subprocess/tool.py - Simple tool for testing subprocess executor

import time
from typing import Dict, Any

from claude_agent_toolkit import BaseTool, tool


class SimpleTool(BaseTool):
    """A simple tool for testing the subprocess executor."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.messages = []
    
    @tool(description="Echo a message back with timestamp")
    async def echo(self, message: str) -> Dict[str, Any]:
        """Echo a message back with additional metadata."""
        self.call_count += 1
        timestamp = time.time()
        
        response_data = {
            "echoed": message,
            "timestamp": timestamp,
            "call_number": self.call_count,
            "message": f"Successfully echoed: '{message}'"
        }
        
        # Store the message for history
        self.messages.append({
            "input": message,
            "timestamp": timestamp,
            "call_number": self.call_count
        })
        
        print(f"\n🔄 [SimpleTool] Echo #{self.call_count}: '{message}'\n")
        
        return response_data
    
    @tool(description="Get the history of echo calls")
    async def get_history(self) -> Dict[str, Any]:
        """Get the history of all echo calls."""
        return {
            "total_calls": self.call_count,
            "messages": self.messages,
            "status": f"Tool has been called {self.call_count} times"
        }
    
    @tool(description="Get simple status information")
    async def status(self) -> Dict[str, Any]:
        """Get current status of the tool."""
        return {
            "active": True,
            "call_count": self.call_count,
            "last_message": self.messages[-1]["input"] if self.messages else None,
            "uptime_info": "Tool is running successfully"
        }