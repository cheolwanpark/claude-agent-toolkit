#!/usr/bin/env python3
# tempfilesys.py - Helper module for creating test filesystem structure

import os
import tempfile
import shutil
from typing import Dict, Any


def create_test_filesystem() -> str:
    """
    Create a temporary directory with test file structure for FileSystemTool testing.
    
    Returns:
        str: Path to the temporary directory
    """
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="fs_tool_test_")
    
    # Create directory structure
    os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "secrets"), exist_ok=True)
    
    # Create test files with sample content
    test_files = get_test_files()
    
    for relative_path, content in test_files.items():
        full_path = os.path.join(test_dir, relative_path)
        
        # Ensure parent directory exists
        parent_dir = os.path.dirname(full_path)
        os.makedirs(parent_dir, exist_ok=True)
        
        # Write file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    return test_dir


def get_test_files() -> Dict[str, str]:
    """
    Get dictionary of test files and their content.
    
    Returns:
        Dict[str, str]: Mapping of relative file paths to content
    """
    return {
        "readme.txt": "This is a readme file.\nIt contains documentation.",
        "data/config.json": '{"setting": "value", "debug": false}',
        "logs/app.log": "INFO: Application started\nWARN: High memory usage\nERROR: Connection failed",
        "secrets/api_key.txt": "secret-api-key-12345",
        "secrets/private.key": "-----BEGIN PRIVATE KEY-----\nsecret-key-data"
    }


def get_test_permissions() -> list:
    """
    Get standard test permission rules.
    
    Returns:
        List[Tuple[str, str]]: Permission rules as (pattern, permission) tuples
    """
    return [
        ("*.txt", "read"),           # Read-only text files
        ("data/**", "write"),        # Write access to data directory
        ("logs/*.log", "read"),      # Read-only log files
        ("data/config.json", "read") # Conflict: write vs read -> resolves to write
    ]


def validate_file_exists(test_dir: str, relative_path: str) -> bool:
    """
    Check if a file exists in the test directory.
    
    Args:
        test_dir: Path to test directory
        relative_path: Relative path to file
        
    Returns:
        bool: True if file exists
    """
    full_path = os.path.join(test_dir, relative_path)
    return os.path.exists(full_path)


def validate_file_content(test_dir: str, relative_path: str, expected_content: str) -> bool:
    """
    Validate that a file has expected content.
    
    Args:
        test_dir: Path to test directory
        relative_path: Relative path to file
        expected_content: Expected file content
        
    Returns:
        bool: True if content matches
    """
    full_path = os.path.join(test_dir, relative_path)
    
    if not os.path.exists(full_path):
        return False
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            actual_content = f.read()
            return actual_content == expected_content
    except Exception:
        return False


def get_file_content(test_dir: str, relative_path: str) -> str:
    """
    Get content of a file in the test directory.
    
    Args:
        test_dir: Path to test directory
        relative_path: Relative path to file
        
    Returns:
        str: File content, or empty string if file doesn't exist
    """
    full_path = os.path.join(test_dir, relative_path)
    
    if not os.path.exists(full_path):
        return ""
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def cleanup_test_directory(test_dir: str) -> None:
    """
    Clean up the test directory.
    
    Args:
        test_dir: Path to test directory to remove
    """
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def print_filesystem_structure(test_dir: str) -> None:
    """
    Print the filesystem structure for debugging.
    
    Args:
        test_dir: Path to test directory
    """
    print(f"📂 Test filesystem structure in {test_dir}:")
    
    for root, dirs, files in os.walk(test_dir):
        # Calculate relative path from test_dir
        rel_root = os.path.relpath(root, test_dir)
        if rel_root == ".":
            rel_root = ""
        
        # Print directories
        for d in dirs:
            if rel_root:
                print(f"  📁 {rel_root}/{d}/")
            else:
                print(f"  📁 {d}/")
        
        # Print files
        for f in files:
            if rel_root:
                print(f"  📄 {rel_root}/{f}")
            else:
                print(f"  📄 {f}")


def validate_agent_results(test_dir: str) -> Dict[str, Any]:
    """
    Validate results after agent operations.
    
    Args:
        test_dir: Path to test directory
        
    Returns:
        Dict[str, Any]: Validation results
    """
    results = {
        "report_created": False,
        "report_content_correct": False,
        "readme_unchanged": False,
        "original_files_intact": True,
        "details": []
    }
    
    # Check if report.txt was created
    if validate_file_exists(test_dir, "data/report.txt"):
        results["report_created"] = True
        results["details"].append("✅ report.txt created in data directory")
        
        # Check if content was updated correctly
        content = get_file_content(test_dir, "data/report.txt")
        if "Final Report" in content:
            results["report_content_correct"] = True
            results["details"].append("✅ report.txt content updated correctly")
        else:
            results["details"].append(f"❌ report.txt content incorrect: {content}")
    else:
        results["details"].append("❌ report.txt was not created")
    
    # Check that readme.txt was not modified
    original_readme = get_test_files()["readme.txt"]
    current_readme = get_file_content(test_dir, "readme.txt")
    
    if current_readme == original_readme:
        results["readme_unchanged"] = True
        results["details"].append("✅ readme.txt unchanged (permission denied worked)")
    else:
        results["details"].append("❌ readme.txt was modified (permission denial failed)")
    
    # Check that original files are intact
    original_files = get_test_files()
    for path, expected_content in original_files.items():
        if path == "readme.txt":  # Already checked above
            continue
            
        current_content = get_file_content(test_dir, path)
        if path == "data/config.json":
            # This file might have been read but should not be modified
            # (it should have write permission due to data/** rule)
            continue
        elif current_content != expected_content:
            results["original_files_intact"] = False
            results["details"].append(f"❌ {path} was modified unexpectedly")
    
    if results["original_files_intact"]:
        results["details"].append("✅ Original files remain intact")
    
    return results