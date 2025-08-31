#!/usr/bin/env python3
# hash_tool.py - Hash computation tool for verification testing

import hashlib
import time
from typing import Dict, Any

from tool import BaseTool, tool


class HashComputerTool(BaseTool):
    """A tool that computes hashes with timestamps."""
    
    def __init__(self):
        super().__init__()
        self.state = {
            "computed_hashes": [],
            "last_hash": None,
            "computation_count": 0
        }
    
    @tool(
        description="Compute SHA256 hash of input with current timestamp",
        cpu_bound=True,
        snapshot=[]
    )
    def compute_hash(self, input_text: str) -> Dict[str, Any]:
        """Compute hash of input combined with timestamp."""
        # Add timestamp to make it unpredictable
        timestamp = str(time.time())
        combined = f"{input_text}:{timestamp}"
        
        # Compute hash
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        # Update state
        self.state["computation_count"] += 1
        self.state["last_hash"] = hash_value
        self.state["computed_hashes"].append({
            "input": input_text,
            "timestamp": timestamp,
            "hash": hash_value,
            "index": self.state["computation_count"]
        })
        
        print(f"\n#️⃣ [Tool] Computed hash #{self.state['computation_count']}: {hash_value[:16]}...\n")
        
        return {
            "hash": hash_value,
            "input": input_text,
            "timestamp": timestamp,
            "message": f"Hash #{self.state['computation_count']} computed"
        }
    
    @tool(
        description="Get the last computed hash",
        cpu_bound=False
    )
    async def get_last_hash(self) -> Dict[str, Any]:
        """Get the most recently computed hash."""
        if self.state["last_hash"] is None:
            return {
                "hash": None,
                "message": "No hash has been computed yet"
            }
        
        last_entry = self.state["computed_hashes"][-1] if self.state["computed_hashes"] else None
        
        return {
            "hash": self.state["last_hash"],
            "computation_count": self.state["computation_count"],
            "details": last_entry
        }
    
    @tool(
        description="Combine two strings and compute their hash",
        cpu_bound=True,
        snapshot=[]
    )
    def combine_and_hash(self, first: str, second: str) -> Dict[str, Any]:
        """Combine two strings and compute hash."""
        timestamp = str(time.time())
        combined = f"{first}|{second}|{timestamp}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        self.state["computation_count"] += 1
        self.state["last_hash"] = hash_value
        self.state["computed_hashes"].append({
            "first": first,
            "second": second,
            "timestamp": timestamp,
            "hash": hash_value,
            "index": self.state["computation_count"]
        })
        
        print(f"\n#️⃣ [Tool] Combined hash #{self.state['computation_count']}: {hash_value[:16]}...\n")
        
        return {
            "hash": hash_value,
            "first": first,
            "second": second,
            "timestamp": timestamp
        }