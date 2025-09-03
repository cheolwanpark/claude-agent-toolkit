#!/usr/bin/env python3
# worker.py - Parallel processing and worker pool management

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

from ..exceptions import ExecutionError


def simple_worker(
    module: str,
    qualname: str, 
    method_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a method in a separate process.
    
    Users are responsible for handling their own race conditions and data management.
    """
    try:
        import importlib
        
        mod = importlib.import_module(module)
        obj = mod
        for part in qualname.split("."):
            obj = getattr(obj, part)
        
        # Create new instance - users manage their own data
        cls = obj
        inst = cls()  # Call normal __init__
        
        # Execute method and return result
        result = getattr(inst, method_name)(*args, **kwargs)
        return {"result": result}
        
    except Exception as e:
        return {"error": str(e)}


class WorkerPoolManager:
    """Manages parallel operation execution with worker pools."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize worker pool manager.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        import os
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        self.cpu_pool: Optional[ProcessPoolExecutor] = None
        self.semaphore: Optional[asyncio.Semaphore] = None
    
    async def execute_parallel(
        self,
        method: Callable,
        meta: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute parallel method in a separate process.
        
        Users are responsible for handling race conditions and data consistency.
        
        Returns:
            Method result directly
            
        Raises:
            ExecutionError: If execution fails or times out
        """
        if self.cpu_pool is None:
            try:
                self.cpu_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            except Exception as e:
                raise ExecutionError(f"Failed to create process pool: {e}") from e
                
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_workers)

        timeout_s: int = meta.get("timeout_s", 60)
        module = method.__module__ or "__main__"
        qualname = method.__qualname__.rsplit('.', 1)[0]  # Remove method name
        method_name = method.__name__

        async with self.semaphore:
            loop = asyncio.get_running_loop()
            try:
                fut = loop.run_in_executor(
                    self.cpu_pool,
                    simple_worker,
                    module, qualname, method_name,
                    args, kwargs
                )
            except Exception as e:
                raise ExecutionError(f"Failed to submit task to process pool: {e}") from e
            
            try:
                result = await asyncio.wait_for(fut, timeout=timeout_s)
            except asyncio.TimeoutError as e:
                raise ExecutionError(f"Operation timed out after {timeout_s}s") from e
            except Exception as e:
                raise ExecutionError(f"Process execution failed: {e}") from e

            if "error" in result:
                raise ExecutionError(f"Parallel operation failed: {result['error']}")

            return result["result"]
    
    def cleanup(self):
        """Clean up worker pool resources."""
        if self.cpu_pool:
            self.cpu_pool.shutdown(wait=False)
            self.cpu_pool = None