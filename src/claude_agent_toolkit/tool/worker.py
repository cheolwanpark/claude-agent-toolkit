#!/usr/bin/env python3
# worker.py - CPU-bound processing and worker pool management

import asyncio
import inspect
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from .state_manager import json_roundtrip, make_patch, apply_patch_inplace
from ..exceptions import ExecutionError, StateError


def autopatch_worker(
    module: str,
    qualname: str,
    method_name: str,
    state0: Any,
    cfg: Dict[str, Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    base_version: int,
) -> Dict[str, Any]:
    """
    Reconstruct a lightweight instance, run the original method mutating a local copy
    of state, compute a JSON Patch, and return (patch, result, base_version).
    """
    import importlib

    mod = importlib.import_module(module)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)

    cls = obj
    inst = object.__new__(cls)          # bypass __init__
    inst.state = deepcopy(state0)       # cloned working state
    for k, v in (cfg or {}).items():    # read-only config fields
        setattr(inst, k, v)

    result = getattr(inst, method_name)(*args, **kwargs)
    state1 = json_roundtrip(inst.state)
    patch = make_patch(state0, state1)
    return {"patch": patch, "result": result, "base_version": base_version}


class WorkerPoolManager:
    """Manages CPU-bound operation execution with worker pools."""
    
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
    
    async def execute_cpu_bound(
        self,
        method: Callable,
        meta: Dict[str, Any],
        state_snapshot: Any,
        current_version: int,
        config_fields: Dict[str, Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute CPU-bound method with retry logic and state management.
        
        Returns:
            Dictionary with 'patch', 'result', and 'base_version' or error info
        """
        if self.cpu_pool is None:
            try:
                self.cpu_pool = ProcessPoolExecutor(max_workers=self.max_workers)
            except Exception as e:
                return {"error": f"Failed to create process pool: {e}"}
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.max_workers)

        timeout_s: int = meta.get("timeout_s", 60)
        policy: str = meta.get("conflict_policy", "retry")
        max_retries: int = int(meta.get("max_retries", 16))
        bo0: int = int(meta.get("backoff_initial_ms", 5))
        bo_max: int = int(meta.get("backoff_max_ms", 250))

        module = method.__module__ or "__main__"
        qualname = method.__qualname__.rsplit('.', 1)[0]  # Remove method name
        method_name = method.__name__

        async with self.semaphore:
            attempt = 0
            while True:
                loop = asyncio.get_running_loop()
                try:
                    fut = loop.run_in_executor(
                        self.cpu_pool,
                        autopatch_worker,
                        module, qualname, method_name,
                        state_snapshot, config_fields,
                        args, kwargs,
                        current_version,
                    )
                except Exception as e:
                    return {"error": f"Failed to submit task to process pool: {e}"}
                
                try:
                    out = await asyncio.wait_for(fut, timeout=timeout_s)
                except asyncio.TimeoutError:
                    return {"error": f"timeout after {timeout_s}s"}
                except Exception as e:
                    return {"error": f"Process execution failed: {e}"}

                if out["base_version"] != current_version:
                    # conflict → retry or surface error
                    if policy == "retry" and attempt < max_retries:
                        delay_ms = min(bo_max, bo0 * (2 ** attempt))
                        # jitter: 50–100% of delay
                        delay = (delay_ms / 1000.0) * (0.5 + random.random() * 0.5)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    return {"error": "conflict", "expected": out["base_version"], "actual": current_version}

                return out
    
    def cleanup(self):
        """Clean up worker pool resources."""
        if self.cpu_pool:
            self.cpu_pool.shutdown(wait=False)
            self.cpu_pool = None