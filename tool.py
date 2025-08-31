#!/usr/bin/env python3
# tool.py - Framework module for MCP tools with state management

import asyncio
import inspect
import json
import os
import random
import socket
import threading
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
import jsonpatch
from concurrent.futures import ProcessPoolExecutor
from fastmcp import FastMCP


# ---------- Utilities (JSON roundtrip + patch) ----------
def json_roundtrip(obj: Any) -> Any:
    return json.loads(json.dumps(obj))

def make_patch(before: Any, after: Any) -> List[dict]:
    return jsonpatch.make_patch(before, after).patch

def apply_patch_inplace(state: Any, patch_ops: List[dict]) -> Any:
    return jsonpatch.apply_patch(state, patch_ops, in_place=True)


# ---------- Worker trampoline (top-level & picklable) ----------
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


# ---------- Public decorator ----------
def tool(
    name: Optional[str] = None,
    description: str = "",
    *,
    cpu_bound: bool = False,
    timeout_s: int = 60,
    snapshot: Optional[List[str]] = None,
    conflict_policy: str = "retry",       # "retry" | "error"
    max_retries: int = 16,                # transparent retries on conflict
    backoff_initial_ms: int = 5,          # exp backoff start
    backoff_max_ms: int = 250,            # cap
):
    def deco(fn: Callable):
        setattr(fn, "__mcp_tool__", True)
        setattr(fn, "__mcp_meta__", {
            "name": name or fn.__name__,
            "description": description,
            "cpu_bound": cpu_bound,
            "timeout_s": timeout_s,
            "snapshot": snapshot or [],
            "conflict_policy": conflict_policy,
            "max_retries": max_retries,
            "backoff_initial_ms": backoff_initial_ms,
            "backoff_max_ms": backoff_max_ms,
        })
        return fn
    return deco


# ---------- BaseTool (server authority + Streamable HTTP) ----------
class BaseTool:
    def __init__(self, *args, **kwargs):
        self._host: str = "127.0.0.1"
        self._port: Optional[int] = None
        self._server_thread: Optional[threading.Thread] = None
        self._ready: bool = False

        # Authoritative mutable state (JSON-serializable)
        self.state: Any = {}
        self._version: int = 0

        # Parallelism bits (one pool per tool instance)
        self._pool_workers: int = max(1, (os.cpu_count() or 2) - 1)
        self._cpu_pool: Optional[ProcessPoolExecutor] = None
        self._sema: Optional[asyncio.Semaphore] = None

        self._init_args = args
        self._init_kwargs = kwargs

    # ---- plumbing ----
    def _pick_port(self) -> int:
        s = socket.socket(); s.bind((self._host, 0))
        port = s.getsockname()[1]; s.close()
        return port

    def _collect(self):
        for _name, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(member, "__mcp_tool__", False):
                meta = getattr(member, "__mcp_meta__", {})
                yield member, meta  # bound method + metadata

    @property
    def connection_url(self) -> str:
        if not self._port:
            raise RuntimeError("Tool is not running")
        return f"http://{self._host}:{self._port}/mcp"  # no trailing slash

    @property
    def health_url(self) -> str:
        if not self._port:
            raise RuntimeError("Tool is not running")
        return f"http://{self._host}:{self._port}/health"

    def _wait_ready(self, timeout: float = 10.0):
        start = time.time(); last_err = None
        while time.time() - start < timeout:
            try:
                r = httpx.get(self.health_url, timeout=0.5)
                if r.status_code == 200:
                    self._ready = True; return
            except Exception as e:
                last_err = e
            time.sleep(0.1)
        raise TimeoutError(f"MCP server not ready at {self.health_url} ({last_err})")

    # ---- CPU-bound method registration (autopatch + retries) ----
    def _register_cpu_tool(self, mcp: FastMCP, method: Callable, meta: Dict[str, Any]):
        sig = inspect.signature(method)
        params = [str(p) for p in sig.parameters.values()]
        param_list = ", ".join(params)
        param_names = [p.name for p in sig.parameters.values()]

        snapshot_cfg_fields: List[str] = meta.get("snapshot", [])
        timeout_s: int = meta.get("timeout_s", 60)
        policy: str = meta.get("conflict_policy", "retry")
        max_retries: int = int(meta.get("max_retries", 16))
        bo0: int = int(meta.get("backoff_initial_ms", 5))
        bo_max: int = int(meta.get("backoff_max_ms", 250))

        module = self.__class__.__module__
        qualname = self.__class__.__qualname__
        method_name = method.__name__

        async def __dispatcher__(**kw):
            if self._cpu_pool is None:
                self._cpu_pool = ProcessPoolExecutor(max_workers=self._pool_workers)
            if self._sema is None:
                self._sema = asyncio.Semaphore(self._pool_workers)

            async with self._sema:
                attempt = 0
                while True:
                    # snapshot state + config
                    state0 = json_roundtrip(self.state)
                    cfg = {k: getattr(self, k) for k in snapshot_cfg_fields}
                    base_ver = self._version

                    loop = asyncio.get_running_loop()
                    fut = loop.run_in_executor(
                        self._cpu_pool,
                        autopatch_worker,
                        module, qualname, method_name,
                        state0, cfg,
                        tuple(kw[p] for p in param_names), {},
                        base_ver,
                    )
                    try:
                        out = await asyncio.wait_for(fut, timeout=timeout_s)
                    except asyncio.TimeoutError:
                        return {"error": f"timeout after {timeout_s}s"}

                    if out["base_version"] != self._version:
                        # conflict → retry or surface error
                        if policy == "retry" and attempt < max_retries:
                            delay_ms = min(bo_max, bo0 * (2 ** attempt))
                            # jitter: 50–100% of delay
                            delay = (delay_ms / 1000.0) * (0.5 + random.random() * 0.5)
                            await asyncio.sleep(delay)
                            attempt += 1
                            continue
                        return {"error": "conflict", "expected": out["base_version"], "actual": self._version}

                    # commit patch
                    apply_patch_inplace(self.state, out["patch"])
                    self._version += 1
                    return out["result"]

        # generate wrapper with SAME signature as original
        ns: Dict[str, Any] = {"__dispatcher__": __dispatcher__}
        code = f"async def _wrapped({param_list}):\n    kw = dict()\n"
        for name in param_names:
            code += f"    kw['{name}'] = {name}\n"
        code += "    return await __dispatcher__(**kw)\n"
        exec(code, ns)
        wrapped = ns["_wrapped"]
        wrapped.__name__ = method.__name__
        wrapped.__doc__ = method.__doc__
        try:
            wrapped.__annotations__ = method.__func__.__annotations__
        except Exception:
            pass

        mcp.tool(name=meta["name"], description=meta["description"])(wrapped)

    # ---- Start server thread & bind tools ----
    def _run_server_thread(self, host: str, port: int):
        mcp = FastMCP(name=self.__class__.__name__)

        @mcp.custom_route("/health", methods=["GET"])
        async def health(_):
            from starlette.responses import PlainTextResponse
            return PlainTextResponse("OK", status_code=200)

        for method, meta in self._collect():
            if meta.get("cpu_bound", False):
                self._register_cpu_tool(mcp, method, meta)
            else:
                mcp.tool(name=meta["name"], description=meta["description"])(method)

        mcp.run(transport="http", host=host, port=port)

    def run(self, host: str = "127.0.0.1", port: Optional[int] = None, *, workers: Optional[int] = None):
        if self._server_thread:
            raise RuntimeError("Already running")
        self._host = host
        self._port = port or self._pick_port()
        if workers is not None:
            self._pool_workers = max(1, int(workers))

        self._server_thread = threading.Thread(
            target=self._run_server_thread,
            args=(self._host, self._port),
            daemon=True,
        )
        self._server_thread.start()
        self._wait_ready()
        print(f"[tool] {self.__class__.__name__} @ {self.connection_url}", flush=True)
        return self