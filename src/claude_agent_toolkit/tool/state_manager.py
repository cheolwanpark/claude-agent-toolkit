#!/usr/bin/env python3
# state_manager.py - State management utilities for MCP tools

import json
from copy import deepcopy
from typing import Any, List

import jsonpatch

from ..exceptions import StateError


def json_roundtrip(obj: Any) -> Any:
    """Serialize and deserialize object for JSON compatibility."""
    try:
        return json.loads(json.dumps(obj))
    except (TypeError, ValueError) as e:
        raise StateError(f"Object is not JSON serializable: {e}") from e


def make_patch(before: Any, after: Any) -> List[dict]:
    """Create JSON patch between two states."""
    try:
        return jsonpatch.make_patch(before, after).patch
    except Exception as e:
        raise StateError(f"Failed to create patch between states: {e}") from e


def apply_patch_inplace(state: Any, patch_ops: List[dict]) -> Any:
    """Apply JSON patch operations to state in-place."""
    try:
        return jsonpatch.apply_patch(state, patch_ops, in_place=True)
    except (jsonpatch.JsonPatchException, ValueError, KeyError) as e:
        raise StateError(f"Failed to apply patch operations: {e}") from e


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
        try:
            return json_roundtrip(self.state), self.version
        except StateError:
            # Re-raise StateError from json_roundtrip
            raise
        except Exception as e:
            raise StateError(f"Failed to create state snapshot: {e}") from e
    
    def apply_changes(self, patch_ops: List[dict]) -> None:
        """Apply patch operations and increment version."""
        try:
            apply_patch_inplace(self.state, patch_ops)
            self.version += 1
        except StateError:
            # Re-raise StateError from apply_patch_inplace
            raise
        except Exception as e:
            raise StateError(f"Failed to apply state changes: {e}") from e
    
    def clone_state(self) -> Any:
        """Create a deep copy of current state."""
        try:
            return deepcopy(self.state)
        except Exception as e:
            raise StateError(f"Failed to clone state: {e}") from e