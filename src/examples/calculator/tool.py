#!/usr/bin/env python3
# calculator/tool.py - Calculator tool for mathematical operations

import math
from datetime import datetime
from typing import Dict, Any, Union

from claude_agent_toolkit import BaseTool, tool


class CalculatorTool(BaseTool):
    """A comprehensive calculator tool with operation history. Users manage data explicitly."""
    
    def __init__(self):
        super().__init__(workers=2)
        # Explicit data management - no automatic state management
        self.history = []
        self.last_result = None
        self.operation_count = 0
    
    def _record_operation(self, operation: str, result: Union[int, float]) -> None:
        """Record an operation in the history."""
        self.operation_count += 1
        self.last_result = result
        self.history.append({
            "id": self.operation_count,
            "operation": operation,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 50 operations
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    @tool(description="Add two numbers together")
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
    
    @tool(description="Subtract the second number from the first number")
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
    
    @tool(description="Multiply two numbers together")
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
    
    @tool(description="Divide the first number by the second number")
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
    
    @tool(description="Raise the first number to the power of the second number using parallel processing", parallel=True, timeout_s=60)
    def power(self, base: float, exponent: float) -> Dict[str, Any]:
        """Raise base to the power of exponent using parallel processing."""
        result = base ** exponent
        operation = f"{base}^{exponent}"
        
        # Note: In parallel execution, self is a new instance, so we can't record to history
        print(f"\n🧮 [Calculator-Parallel] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Raised {base} to the power of {exponent} to get {result} (calculated in parallel process)",
            "parallel_execution": True
        }
    
    @tool(description="Calculate the square root of a number using parallel processing", parallel=True, timeout_s=30)
    def square_root(self, number: float) -> Dict[str, Any]:
        """Calculate the square root of a number using parallel processing."""
        if number < 0:
            return {
                "error": "Cannot calculate square root of negative number",
                "operation": f"√{number}",
                "result": None
            }
        
        result = math.sqrt(number)
        operation = f"√{number}"
        
        # Note: In parallel execution, self is a new instance, so we can't record to history
        print(f"\n🧮 [Calculator-Parallel] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Square root of {number} is {result} (calculated in parallel process)",
            "parallel_execution": True
        }
    
    @tool(description="Get the last calculation result")
    async def get_last_result(self) -> Dict[str, Any]:
        """Get the result of the last calculation."""
        return {
            "last_result": self.last_result,
            "operation_count": self.operation_count,
            "message": f"Last result: {self.last_result}"
        }
    
    @tool(description="Get the calculation history")
    async def get_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get the recent calculation history."""
        recent_history = self.history[-limit:] if self.history else []
        
        return {
            "history": recent_history,
            "total_operations": self.operation_count,
            "limit": limit,
            "message": f"Retrieved last {len(recent_history)} operations from history"
        }
    
    @tool(description="Clear the calculation history and reset data")
    async def clear_history(self) -> Dict[str, Any]:
        """Clear all calculation history and reset data."""
        self.history = []
        self.last_result = None
        self.operation_count = 0
        
        print(f"\n🧮 [Calculator] History cleared\n")
        
        return {
            "message": "Calculator history has been cleared and data reset",
            "cleared": True
        }

    
    @tool(description="Calculate factorial of a number using parallel processing", parallel=True, timeout_s=120)
    def factorial(self, n: int) -> Dict[str, Any]:
        """Calculate factorial of n using parallel processing for CPU-intensive computation."""
        if n < 0:
            return {
                "error": "Factorial is not defined for negative numbers",
                "operation": f"{n}!",
                "result": None
            }
        
        if n > 20:
            return {
                "error": "Factorial calculation limited to n <= 20 for safety",
                "operation": f"{n}!",
                "result": None
            }
        
        # CPU-intensive computation suitable for parallel processing
        result = math.factorial(n)
        operation = f"{n}!"
        
        # Note: In parallel execution, self is a new instance, so we can't record to history
        # This demonstrates the ProcessPoolExecutor behavior
        print(f"\n🧮 [Calculator-Parallel] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Factorial of {n} is {result} (calculated in parallel process)",
            "parallel_execution": True
        }
    
    @tool(description="Calculate Fibonacci number using parallel processing", parallel=True, timeout_s=120)
    def fibonacci(self, n: int) -> Dict[str, Any]:
        """Calculate the nth Fibonacci number using parallel processing."""
        if n < 0:
            return {
                "error": "Fibonacci sequence is not defined for negative numbers",
                "operation": f"fib({n})",
                "result": None
            }
        
        if n > 35:
            return {
                "error": "Fibonacci calculation limited to n <= 35 for reasonable performance",
                "operation": f"fib({n})",
                "result": None
            }
        
        # Recursive Fibonacci calculation (CPU-intensive)
        def fib(num):
            if num <= 1:
                return num
            return fib(num - 1) + fib(num - 2)
        
        result = fib(n)
        operation = f"fib({n})"
        
        print(f"\n🧮 [Calculator-Parallel] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"Fibonacci number {n} is {result} (calculated in parallel process)",
            "parallel_execution": True
        }
    
    @tool(description="Check if a number is prime using parallel processing", parallel=True, timeout_s=60)
    def is_prime(self, n: int) -> Dict[str, Any]:
        """Check if a number is prime using parallel processing for CPU-intensive computation."""
        if n < 2:
            return {
                "operation": f"is_prime({n})",
                "result": False,
                "message": f"{n} is not prime (numbers less than 2 are not prime)",
                "parallel_execution": True
            }
        
        if n > 1000000:
            return {
                "error": "Prime checking limited to numbers <= 1,000,000 for performance",
                "operation": f"is_prime({n})",
                "result": None
            }
        
        # CPU-intensive prime checking algorithm
        if n == 2:
            result = True
        elif n % 2 == 0:
            result = False
        else:
            result = True
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    result = False
                    break
        
        operation = f"is_prime({n})"
        
        print(f"\n🧮 [Calculator-Parallel] {operation} = {result}\n")
        
        return {
            "operation": operation,
            "result": result,
            "message": f"{n} {'is' if result else 'is not'} a prime number (calculated in parallel process)",
            "parallel_execution": True
        }
