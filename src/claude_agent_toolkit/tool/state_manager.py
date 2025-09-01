#!/usr/bin/env python3
# state_manager.py - State management utilities for MCP tools

import json
from copy import deepcopy
from typing import Any, List

import jsonpatch


def json_roundtrip(obj: Any) -> Any:
    """Serialize and deserialize object for JSON compatibility."""
    return json.loads(json.dumps(obj))


def make_patch(before: Any, after: Any) -> List[dict]:
    """Create JSON patch between two states."""
    return jsonpatch.make_patch(before, after).patch


def apply_patch_inplace(state: Any, patch_ops: List[dict]) -> Any:
    """Apply JSON patch operations to state in-place."""
    return jsonpatch.apply_patch(state, patch_ops, in_place=True)


class StateManager:
    """Manages versioned state for MCP tools."""
    
    def __init__(self, initial_state: Any = None):
        """
        Initialize state manager.
        
        Args:
            initial_state: Initial state value (defaults to empty dict)
        """
        self.state: Any = initial_state if initial_state is not None else {}
        self.version: int = 0
    
    def get_snapshot(self) -> tuple[Any, int]:
        """Get current state snapshot and version."""
        return json_roundtrip(self.state), self.version
    
    def apply_changes(self, patch_ops: List[dict]) -> None:
        """Apply patch operations and increment version."""
        apply_patch_inplace(self.state, patch_ops)
        self.version += 1
    
    def clone_state(self) -> Any:
        """Create a deep copy of current state."""
        return deepcopy(self.state)