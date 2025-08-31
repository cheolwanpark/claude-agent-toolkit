#!/usr/bin/env python3
# hash_example.py - Independent hash computer verification demo

import asyncio
import os

# Absolute imports for independence
from agent import Agent
from examples.hash_tool import HashComputerTool


async def main():
    """Standalone hash computer verification demo."""
    print("\n" + "="*60)
    print("HASH COMPUTER VERIFICATION DEMO")
    print("="*60)
    print("\nThis demo verifies that agents actually call the hash computer tool by:")
    print("1. Using a tool that computes hashes with unpredictable timestamps")
    print("2. Checking tool state to confirm it was called")
    print("3. Verifying the unpredictable hash values appear in agent responses")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        return False
    
    try:
        # Start the hash computer tool
        hash_tool = HashComputerTool().run(workers=2)
        
        # Create agent and connect to tool
        agent = Agent()
        agent.connect(hash_tool)
        
        print("\n📝 Test 1: Compute hash of specific input")
        print("-" * 40)
        
        test_input = "Hello World"
        result = await agent.run(
            f"Please compute the SHA256 hash of '{test_input}' using the compute_hash tool. "
            "Tell me the complete hash value."
        )
        
        print(f"\n[Agent Response]:")
        print(f"Success: {result.get('success')}")
        response = result.get('response', '')
        print(f"Response: {response[:500]}...")
        
        # Verify by checking tool state
        last_hash = hash_tool.state.get("last_hash")
        computation_count = hash_tool.state.get("computation_count", 0)
        
        print(f"\n✅ Verification:")
        print(f"  - Tool was called: {computation_count > 0}")
        print(f"  - Hash computed by tool: {last_hash[:32] if last_hash else None}...")
        print(f"  - Hash appears in response: {last_hash in response if last_hash else False}")
        
        # Test 2: Combine and hash
        print("\n📝 Test 2: Combine strings and hash")
        print("-" * 40)
        
        result = await agent.run(
            "Please use the combine_and_hash tool to combine 'Alice' and 'Bob' "
            "and give me the resulting hash."
        )
        
        print(f"\n[Agent Response]:")
        print(f"Response: {result.get('response', '')[:300]}...")
        
        final_count = hash_tool.state.get("computation_count", 0)
        print(f"\n✅ Final computation count: {final_count}")
        
        # Final verification
        success = computation_count > 0 and last_hash is not None
        
        print("\n" + "="*60)
        print("HASH DEMO SUMMARY")
        print("="*60)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Hash Computer Verification: {status}")
        
        if success:
            print("\n🎉 Hash verification demo passed! The agent called the tool correctly.")
        else:
            print("\n⚠️  Demo failed. The agent may not be calling the tool properly.")
        
        return success
        
    except RuntimeError as e:
        if "Cannot connect to Docker" in str(e):
            print(f"\n{e}")
            print("\n💡 After starting Docker, run this demo again.")
        else:
            raise
        return False
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n" + "="*60)
        print("HASH DEMO COMPLETED")
        print("="*60)


if __name__ == "__main__":
    # Run the standalone demo
    asyncio.run(main())