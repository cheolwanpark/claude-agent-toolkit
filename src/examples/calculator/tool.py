#!/usr/bin/env python3
# calculator/tool.py - Calculator tool for mathematical operations

import math
from datetime import datetime
from typing import Dict, Any, Union

from claude_agent_toolkit import BaseTool, tool


class CalculatorTool(BaseTool):
    """A comprehensive calculator tool with operation history and state management."""
    
    def __init__(self):
        super().__init__()
        self.state = {
            "history": [],
            "last_result": None,
            "operation_count": 0
        }
    
    def _record_operation(self, operation: str, result: Union[int, float]) -> None:
        """Record an operation in the history."""
        self.state["operation_count"] += 1
        self.state["last_result"] = result
        self.state["history"].append({
            "id": self.state["operation_count"],
            "operation": operation,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 50 operations
        if len(self.state["history"]) > 50:
            self.state["history"] = self.state["history"][-50:]
    
    @tool(
        description="Add two numbers together",
        cpu_bound=False
    )
    async def add(self, a: float, b: float) -> Dict[str, Any]:
        """Add two numbers and return the result."""
        result = a + b
        operation = f"{a} + {b}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Added {a} and {b} to get {result}"
        }
    
    @tool(
        description="Subtract the second number from the first number",
        cpu_bound=False
    )
    async def subtract(self, a: float, b: float) -> Dict[str, Any]:
        """Subtract b from a and return the result."""
        result = a - b
        operation = f"{a} - {b}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Subtracted {b} from {a} to get {result}"
        }
    
    @tool(
        description="Multiply two numbers together",
        cpu_bound=False
    )
    async def multiply(self, a: float, b: float) -> Dict[str, Any]:
        """Multiply two numbers and return the result."""
        result = a * b
        operation = f"{a} × {b}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Multiplied {a} and {b} to get {result}"
        }
    
    @tool(
        description="Divide the first number by the second number",
        cpu_bound=False
    )
    async def divide(self, a: float, b: float) -> Dict[str, Any]:
        """Divide a by b and return the result."""
        if b == 0:
            return {
                "error": "Division by zero is not allowed",
                "operation": f"{a} ÷ {b}",
                "result": None
            }
        
        result = a / b
        operation = f"{a} ÷ {b}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Divided {a} by {b} to get {result}"
        }
    
    @tool(
        description="Raise the first number to the power of the second number",
        cpu_bound=False
    )
    async def power(self, base: float, exponent: float) -> Dict[str, Any]:
        """Raise base to the power of exponent."""
        result = base ** exponent
        operation = f"{base}^{exponent}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Raised {base} to the power of {exponent} to get {result}"
        }
    
    @tool(
        description="Calculate the square root of a number",
        cpu_bound=False
    )
    async def square_root(self, number: float) -> Dict[str, Any]:
        """Calculate the square root of a number."""
        if number < 0:
            return {
                "error": "Cannot calculate square root of negative number",
                "operation": f"√{number}",
                "result": None
            }
        
        result = math.sqrt(number)
        operation = f"√{number}"
        self._record_operation(operation, result)
        
        print(f"\n🧮 [Calculator] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Square root of {number} is {result}"
        }
    
    @tool(
        description="Get the last calculation result",
        cpu_bound=False
    )
    async def get_last_result(self) -> Dict[str, Any]:
        """Get the result of the last calculation."""
        return {
            "last_result": self.state["last_result"],
            "operation_count": self.state["operation_count"],
            "message": f"Last result: {self.state['last_result']}"
        }
    
    @tool(
        description="Get the calculation history",
        cpu_bound=False
    )
    async def get_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get the recent calculation history."""
        recent_history = self.state["history"][-limit:] if self.state["history"] else []
        
        return {
            "history": recent_history,
            "total_operations": self.state["operation_count"],
            "limit": limit,
            "message": f"Retrieved last {len(recent_history)} operations from history"
        }
    
    @tool(
        description="Clear the calculation history and reset state",
        cpu_bound=False
    )
    async def clear_history(self) -> Dict[str, Any]:
        """Clear all calculation history and reset state."""
        self.state = {
            "history": [],
            "last_result": None,
            "operation_count": 0
        }
        
        print(f"\n🧮 [Calculator] History cleared\n")
        
        return {
            "message": "Calculator history has been cleared and state reset",
            "cleared": True
        }