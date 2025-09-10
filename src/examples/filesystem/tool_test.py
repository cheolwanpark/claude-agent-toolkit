#!/usr/bin/env python3
# tool_test.py - Direct FileSystemTool unit tests

import asyncio
from claude_agent_toolkit.tools import FileSystemTool
from tempfilesys import create_test_filesystem, get_test_permissions, cleanup_test_directory, print_filesystem_structure


def run_permission_resolution_tests():
    """Test permission resolution logic directly."""
    print("🔍 Testing permission resolution...")
    
    permissions = get_test_permissions()
    fs_tool = FileSystemTool(permissions=permissions, root_dir="/tmp")
    
    # Test cases: (path, expected_permission)
    test_cases = [
        ("readme.txt", "read"),
        ("secrets/api_key.txt", "read"),  # matches *.txt
        ("data/new_file.txt", "write"),   # matches data/** (write wins over *.txt read)
        ("data/config.json", "write"),    # conflict: data/** write vs data/config.json read -> write wins
        ("logs/app.log", "read"),
        ("secrets/private.key", None),    # no matching pattern
        ("data/subdir/file.txt", "write"), # matches data/**
    ]
    
    for path, expected in test_cases:
        actual = fs_tool._resolve_permission(path)
        status = "✅" if actual == expected else "❌"
        print(f"  {status} {path}: expected {expected}, got {actual}")
        
        if actual != expected:
            raise AssertionError(f"Permission resolution failed for {path}")
    
    print("✅ Permission resolution tests passed")


async def run_file_operation_tests():
    """Test file operations with real filesystem."""
    print("\n📁 Testing file operations...")
    
    # Create test filesystem
    test_dir = create_test_filesystem()
    print(f"Created test directory: {test_dir}")
    
    try:
        # Create filesystem tool
        permissions = get_test_permissions()
        fs_tool = FileSystemTool(permissions=permissions, root_dir=test_dir)
        
        # Test 1: List files
        print("\n📋 Test 1: Listing files")
        result = await fs_tool.list()
        print(f"  Found {result['total_files']} files, {result['total_directories']} directories")
        assert result['total_files'] >= 4, "Should find at least 4 files"
        assert 'readme.txt' in result['files'], "Should find readme.txt"
        assert result['files']['readme.txt'] == 'read', "readme.txt should be read-only"
        assert result['files']['data/config.json'] == 'write', "config.json should have write permission"
        print("  ✅ List test passed")
        
        # Test 2: Read operations
        print("\n📖 Test 2: Read operations")
        
        # Should work: read readme.txt
        result = await fs_tool.read("readme.txt")
        assert 'content' in result, "Should be able to read readme.txt"
        assert "documentation" in result['content'], "Should contain expected content"
        print("  ✅ Read readme.txt: Success")
        
        # Should work: read secrets/api_key.txt (matches *.txt)
        result = await fs_tool.read("secrets/api_key.txt")
        assert 'content' in result, "Should be able to read api_key.txt"
        print("  ✅ Read secrets/api_key.txt: Success")
        
        # Should fail: read secrets/private.key (no matching pattern)
        result = await fs_tool.read("secrets/private.key")
        assert 'error' in result, "Should fail to read private.key"
        assert 'Permission denied' in result['error'], "Should be permission error"
        print("  ✅ Read secrets/private.key: Correctly denied")
        
        # Test 3: Write operations
        print("\n✍️  Test 3: Write operations")
        
        # Should work: write to data directory
        result = await fs_tool.write("data/new_file.txt", "This is test content")
        assert result.get('success'), f"Should be able to write to data/: {result.get('error')}"
        print("  ✅ Write data/new_file.txt: Success")
        
        # Should fail: write to readme.txt (read-only)
        result = await fs_tool.write("readme.txt", "New content")
        assert not result.get('success'), "Should fail to write to readme.txt"
        assert 'Write permission denied' in result['error'], "Should be permission error"
        print("  ✅ Write readme.txt: Correctly denied")
        
        # Test 4: Update operations
        print("\n🔄 Test 4: Update operations")
        
        # Should work: update file in data directory
        result = await fs_tool.update("data/new_file.txt", "test", "updated")
        assert result.get('success'), f"Should be able to update data file: {result.get('error')}"
        assert result.get('replacements') == 1, "Should replace one occurrence"
        print("  ✅ Update data/new_file.txt: Success")
        
        # Verify the update worked
        result = await fs_tool.read("data/new_file.txt")
        assert 'updated' in result['content'], "Content should be updated"
        print("  ✅ Update verification: Content changed")
        
        # Should fail: update read-only file
        result = await fs_tool.update("readme.txt", "readme", "changed")
        assert not result.get('success'), "Should fail to update readme.txt"
        print("  ✅ Update readme.txt: Correctly denied")
        
        # Test 5: Security features
        print("\n🛡️  Test 5: Security features")
        
        # Path traversal attack
        result = await fs_tool.read("../../../etc/passwd")
        assert 'error' in result, "Should block path traversal"
        assert 'Invalid path' in result['error'], "Should be invalid path error"
        print("  ✅ Path traversal: Blocked")
        
        # Test 6: Conflict resolution
        print("\n⚖️  Test 6: Permission conflict resolution")
        
        # data/config.json matches both "data/**" (write) and "data/config.json" (read)
        # Should resolve to write (more permissive)
        result = await fs_tool.write("data/config.json", "New config content")
        assert result.get('success'), "Conflict should resolve to write permission"
        print("  ✅ Conflict resolution: Write permission won")
        
        print("\n✅ All file operation tests passed")
        
    finally:
        cleanup_test_directory(test_dir)
        print(f"Cleaned up test directory: {test_dir}")


async def run_tool_tests():
    """Run all tool tests."""
    print("🧪 FileSystemTool Unit Tests")
    print("=" * 50)
    
    # Test 1: Permission resolution (sync)
    run_permission_resolution_tests()
    
    # Test 2: File operations (async)
    await run_file_operation_tests()
    
    print("\n" + "=" * 50)
    print("✅ All tool tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_tool_tests())