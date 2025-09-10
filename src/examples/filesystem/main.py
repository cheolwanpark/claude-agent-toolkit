#!/usr/bin/env python3
# filesystem/main.py - FileSystemTool demo with Agent integration

import asyncio
import os
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ConnectionError, ConfigurationError, ExecutionError
from claude_agent_toolkit.tools import FileSystemTool

# Import our helpers and prompts
from tempfilesys import (
    create_test_filesystem, 
    get_test_permissions, 
    cleanup_test_directory,
    validate_agent_results,
    print_filesystem_structure
)
from prompt import FILESYSTEM_SYSTEM_PROMPT
from tool_test import run_tool_tests


async def run_agent_test():
    """Run agent test with FileSystemTool."""
    print("\n" + "=" * 60)
    print("AGENT TEST - FileSystemTool Integration")
    print("=" * 60)
    
    # Create test filesystem
    test_dir = create_test_filesystem()
    print(f"📂 Created test directory: {test_dir}")
    
    try:
        # Show initial filesystem structure
        print_filesystem_structure(test_dir)
        
        # Create FileSystemTool with test permissions
        permissions = get_test_permissions()
        print(f"\n🔐 Permission rules:")
        for pattern, perm in permissions:
            print(f"  - {pattern}: {perm}")
        
        fs_tool = FileSystemTool(permissions=permissions, root_dir=test_dir)
        
        # Create agent with FileSystemTool
        agent = Agent(
            system_prompt=FILESYSTEM_SYSTEM_PROMPT,
            tools=[fs_tool]
        )
        
        print(f"\n🤖 Running agent test...")
        print("-" * 40)
        
        # Simple instruction prompt (no interaction)
        instruction = """Please perform these filesystem operations:

1. List all files and show their permissions
2. Read the config.json file from the data directory  
3. Write a new file called 'report.txt' in the data directory with content 'Test Report'
4. Update the report.txt file to replace 'Test' with 'Final'
5. Try to write to readme.txt (this should fail due to permissions)

Report what you did for each step and any errors you encountered."""
        
        # Run agent with instructions
        response = await agent.run(instruction)
        
        print(f"\n[Agent Response]:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Validate that agent actually performed the operations
        print(f"\n🔍 Validating agent results...")
        validation_results = validate_agent_results(test_dir)
        
        # Print validation details
        for detail in validation_results["details"]:
            print(f"  {detail}")
        
        # Check overall success
        success = (
            validation_results["report_created"] and
            validation_results["report_content_correct"] and
            validation_results["readme_unchanged"] and
            validation_results["original_files_intact"]
        )
        
        if success:
            print(f"\n✅ Agent test PASSED - All operations performed correctly!")
            return True
        else:
            print(f"\n❌ Agent test FAILED - Some operations were not performed correctly")
            return False
            
    except (ConfigurationError, ConnectionError, ExecutionError) as e:
        print(f"\n❌ Agent Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_directory(test_dir)
        print(f"\n🧹 Cleaned up test directory: {test_dir}")


async def main():
    """Main entry point for filesystem demo."""
    print("\n" + "="*60)
    print("CLAUDE AGENT TOOLKIT - FILESYSTEM DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("1. FileSystemTool implementation with pattern-based permissions")
    print("2. Direct tool testing (unit tests)")
    print("3. Agent integration with filesystem operations")
    print("4. Validation of actual file system changes")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        print("\nRunning tool tests only (no agent test)...")
        
        # Run tool tests without agent
        await run_tool_tests()
        return True
    
    try:
        # Step 1: Run direct tool unit tests
        print(f"\n" + "=" * 60)
        print("STEP 1: TOOL UNIT TESTS")
        print("=" * 60)
        
        await run_tool_tests()
        
        # Step 2: Run agent integration test
        print(f"\n" + "=" * 60)
        print("STEP 2: AGENT INTEGRATION TEST")
        print("=" * 60)
        
        agent_success = await run_agent_test()
        
        # Final results
        print(f"\n" + "=" * 60)
        print("FILESYSTEM DEMO RESULTS")
        print("=" * 60)
        print("✅ Tool unit tests: PASSED")
        print(f"{'✅' if agent_success else '❌'} Agent integration test: {'PASSED' if agent_success else 'FAILED'}")
        
        if agent_success:
            print("\n🎉 All tests passed! FileSystemTool is working correctly.")
            return True
        else:
            print("\n⚠️  Agent test failed, but tool tests passed.")
            return False
            
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        if "Docker" in str(e):
            print("\n💡 Please start Docker Desktop and run this demo again.")
        elif "bind" in str(e) or "port" in str(e):
            print("\n💡 Port may be in use. Try again in a moment.")
        return False
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Check your OAuth token and tool configuration.")
        return False
    except ExecutionError as e:
        print(f"\n❌ Execution Error: {e}")
        print("\n💡 The agent execution failed. Check the error details above.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during filesystem demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "=" * 60)
        print("FILESYSTEM DEMO COMPLETED")
        print("=" * 60)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)