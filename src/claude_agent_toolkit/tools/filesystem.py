#!/usr/bin/env python3
# filesystem.py - Filesystem tool with pattern-based permissions

import os
import fnmatch
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from claude_agent_toolkit import BaseTool, tool

# Type definitions
PermissionLevel = str  # 'read' or 'write'
PermissionRule = Tuple[str, PermissionLevel]  # (pattern, permission)


class FileSystemTool(BaseTool):
    """
    A filesystem access tool with pattern-based permissions.
    
    Features:
    - Pattern-based file/directory selection using glob patterns
    - Granular permission system (read, write where write includes read)
    - Strict permission conflict resolution (defaults to read-only)
    - Security containment within root directory
    """
    
    def __init__(self, permissions: List[PermissionRule], root_dir: str = "."):
        """
        Initialize filesystem tool with permission rules.
        
        Args:
            permissions: List of (pattern, permission) tuples
                        e.g., [('*.txt', 'read'), ('data/**', 'write')]
            root_dir: Root directory for all operations (security boundary)
        """
        super().__init__()
        
        # Normalize and store root directory
        self._root_dir = os.path.abspath(root_dir)
        
        # Store permission rules
        self._permissions = permissions
        
        # Validate permissions
        self._validate_permissions()
    
    def _validate_permissions(self) -> None:
        """Validate permission rules format."""
        for pattern, permission in self._permissions:
            if not isinstance(pattern, str):
                raise ValueError(f"Pattern must be string, got {type(pattern)}")
            if permission not in ['read', 'write']:
                raise ValueError(f"Permission must be 'read' or 'write', got '{permission}'")
    
    def _normalize_path(self, path: str) -> Optional[str]:
        """
        Normalize path and ensure it's within root directory.
        
        Args:
            path: Input path (relative or absolute)
            
        Returns:
            Normalized absolute path within root_dir, or None if invalid
        """
        try:
            # Handle relative paths
            if not os.path.isabs(path):
                abs_path = os.path.abspath(os.path.join(self._root_dir, path))
            else:
                abs_path = os.path.abspath(path)
            
            # Ensure path is within root directory
            if not abs_path.startswith(self._root_dir + os.sep) and abs_path != self._root_dir:
                return None
                
            return abs_path
            
        except (ValueError, OSError):
            return None
    
    def _get_relative_path(self, abs_path: str) -> str:
        """Convert absolute path to relative path from root_dir."""
        return os.path.relpath(abs_path, self._root_dir)
    
    def _resolve_permission(self, path: str) -> Optional[PermissionLevel]:
        """
        Resolve permission for a given path using permission rules.
        
        Args:
            path: Relative path from root_dir
            
        Returns:
            'read', 'write', or None (denied)
        """
        matched_permissions = []
        
        # Check each permission rule
        for pattern, permission in self._permissions:
            if fnmatch.fnmatch(path, pattern):
                matched_permissions.append(permission)
        
        # No matches = denied
        if not matched_permissions:
            return None
        
        # Conflict resolution: if any rule grants 'write', result is 'write' (more permissive)
        if 'write' in matched_permissions:
            return 'write'
        
        # Only 'read' permissions = read
        return 'read'
    
    def _list_directory_recursive(self, dir_path: str) -> List[str]:
        """Recursively list all files and directories."""
        items = []
        try:
            for root, dirs, files in os.walk(dir_path):
                # Add directories
                for d in dirs:
                    full_path = os.path.join(root, d)
                    rel_path = self._get_relative_path(full_path)
                    items.append(rel_path + '/')
                
                # Add files
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = self._get_relative_path(full_path)
                    items.append(rel_path)
        except OSError:
            pass  # Skip directories we can't read
        
        return items
    
    @tool(description="Lists all accessible files and directories with their permissions")
    async def list(self) -> Dict[str, Any]:
        """
        List all files and directories with their effective permissions.
        
        Returns:
            Dict with 'files' and 'directories' containing path->permission mappings
        """
        try:
            # Get all items in root directory
            all_items = self._list_directory_recursive(self._root_dir)
            
            files = {}
            directories = {}
            
            for item in all_items:
                permission = self._resolve_permission(item)
                
                if permission:  # Only include accessible items
                    if item.endswith('/'):
                        # Directory
                        directories[item] = permission
                    else:
                        # File
                        files[item] = permission
            
            return {
                'files': files,
                'directories': directories,
                'total_files': len(files),
                'total_directories': len(directories),
                'message': f'Found {len(files)} accessible files and {len(directories)} accessible directories'
            }
            
        except Exception as e:
            return {
                'error': f'Failed to list files: {str(e)}',
                'files': {},
                'directories': {}
            }
    
    @tool(description="Reads and returns the content of a file")
    async def read(self, filename: str) -> Dict[str, Any]:
        """
        Read file content.
        
        Args:
            filename: Path to file relative to root_dir
            
        Returns:
            Dict with file content or error message
        """
        # Normalize path
        abs_path = self._normalize_path(filename)
        if abs_path is None:
            return {
                'error': f'Invalid path: {filename}',
                'content': None
            }
        
        rel_path = self._get_relative_path(abs_path)
        
        # Check permission
        permission = self._resolve_permission(rel_path)
        if permission is None:
            return {
                'error': f'Permission denied: {filename}',
                'content': None
            }
        
        # Both 'read' and 'write' permissions allow reading
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                'content': content,
                'filename': filename,
                'size': len(content),
                'permission': permission,
                'message': f'Successfully read {filename} ({len(content)} characters)'
            }
            
        except FileNotFoundError:
            return {
                'error': f'File not found: {filename}',
                'content': None
            }
        except PermissionError:
            return {
                'error': f'System permission denied: {filename}',
                'content': None
            }
        except UnicodeDecodeError:
            return {
                'error': f'Cannot decode file as UTF-8: {filename}',
                'content': None
            }
        except Exception as e:
            return {
                'error': f'Failed to read {filename}: {str(e)}',
                'content': None
            }
    
    @tool(description="Writes content to a file, creating or overwriting as needed")
    async def write(self, filename: str, content: str) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            filename: Path to file relative to root_dir
            content: Content to write
            
        Returns:
            Dict with success message or error
        """
        # Normalize path
        abs_path = self._normalize_path(filename)
        if abs_path is None:
            return {
                'error': f'Invalid path: {filename}',
                'success': False
            }
        
        rel_path = self._get_relative_path(abs_path)
        
        # Check permission - need 'write' permission
        permission = self._resolve_permission(rel_path)
        if permission != 'write':
            return {
                'error': f'Write permission denied: {filename}',
                'success': False
            }
        
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(abs_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            # Write file
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'filename': filename,
                'size': len(content),
                'message': f'Successfully wrote {len(content)} characters to {filename}'
            }
            
        except PermissionError:
            return {
                'error': f'System permission denied: {filename}',
                'success': False
            }
        except Exception as e:
            return {
                'error': f'Failed to write {filename}: {str(e)}',
                'success': False
            }
    
    @tool(description="Updates file content by replacing original string with updated string")
    async def update(self, filename: str, original: str, update: str) -> Dict[str, Any]:
        """
        Update file by replacing original string with updated string.
        
        Args:
            filename: Path to file relative to root_dir
            original: String to find and replace
            update: Replacement string
            
        Returns:
            Dict with replacement count or error message
        """
        # Normalize path
        abs_path = self._normalize_path(filename)
        if abs_path is None:
            return {
                'error': f'Invalid path: {filename}',
                'success': False,
                'replacements': 0
            }
        
        rel_path = self._get_relative_path(abs_path)
        
        # Check permission - need 'write' permission
        permission = self._resolve_permission(rel_path)
        if permission != 'write':
            return {
                'error': f'Write permission denied: {filename}',
                'success': False,
                'replacements': 0
            }
        
        try:
            # Read current content
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if original string exists
            if original not in content:
                return {
                    'error': f'Original string not found in {filename}',
                    'success': False,
                    'replacements': 0
                }
            
            # Perform replacement
            new_content = content.replace(original, update)
            replacement_count = content.count(original)
            
            # Write updated content
            with open(abs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {
                'success': True,
                'filename': filename,
                'replacements': replacement_count,
                'original': original,
                'update': update,
                'message': f'Successfully replaced {replacement_count} occurrence(s) in {filename}'
            }
            
        except FileNotFoundError:
            return {
                'error': f'File not found: {filename}',
                'success': False,
                'replacements': 0
            }
        except PermissionError:
            return {
                'error': f'System permission denied: {filename}',
                'success': False,
                'replacements': 0
            }
        except UnicodeDecodeError:
            return {
                'error': f'Cannot decode file as UTF-8: {filename}',
                'success': False,
                'replacements': 0
            }
        except Exception as e:
            return {
                'error': f'Failed to update {filename}: {str(e)}',
                'success': False,
                'replacements': 0
            }