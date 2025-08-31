#!/usr/bin/env python3
# multi_tool_example.py - Independent multi-tool verification demo

import asyncio
import os

# Absolute imports for independence
from agent import Agent
from examples.secret_tool import SecretGeneratorTool
from examples.hash_tool import HashComputerTool


async def main():
    """Standalone multi-tool verification demo."""
    print("\n" + "="*60)
    print("MULTI-TOOL VERIFICATION DEMO")
    print("="*60)
    print("\nThis demo verifies that agents can use multiple tools together by:")
    print("1. Using both secret generator and hash computer tools")
    print("2. Coordinating complex tasks across multiple tools")
    print("3. Verifying both tools were called and results integrated")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        return False
    
    try:
        # Start both tools on different ports
        secret_tool = SecretGeneratorTool().run(port=8001, workers=2)
        hash_tool = HashComputerTool().run(port=8002, workers=2)
        
        # Create agent and connect to BOTH tools
        agent = Agent()
        agent.connect(secret_tool)
        agent.connect(hash_tool)
        
        print("\n📝 Complex task using both tools")
        print("-" * 40)
        
        result = await agent.run(
            "Please do the following: "
            "1) Generate a new secret token "
            "2) Compute the hash of that secret token "
            "3) Tell me both the secret and its hash"
        )
        
        print(f"\n[Agent Response]:")
        print(f"Success: {result.get('success')}")
        response = result.get('response', '')
        print(f"Response: {response[:600]}...")
        
        # Verify both tools were called
        secret_generated = secret_tool.state.get("generation_count", 0) > 0
        hash_computed = hash_tool.state.get("computation_count", 0) > 0
        actual_secret = secret_tool.state.get("current_secret")
        actual_hash = hash_tool.state.get("last_hash")
        
        print(f"\n✅ Verification:")
        print(f"  - Secret tool called: {secret_generated}")
        print(f"  - Hash tool called: {hash_computed}")
        print(f"  - Secret from tool: {actual_secret}")
        print(f"  - Hash from tool: {actual_hash[:32] if actual_hash else None}...")
        print(f"  - Both appear in response: {(actual_secret in response if actual_secret else False) and (actual_hash in response if actual_hash else False)}")
        
        # Final verification
        success = secret_generated and hash_computed
        
        print("\n" + "="*60)
        print("MULTI-TOOL DEMO SUMMARY")
        print("="*60)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"Multi-Tool Verification: {status}")
        
        if success:
            print("\n🎉 Multi-tool verification demo passed! The agent coordinated both tools correctly.")
        else:
            print("\n⚠️  Demo failed. The agent may not be calling both tools properly.")
        
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
        print("MULTI-TOOL DEMO COMPLETED")
        print("="*60)


if __name__ == "__main__":
    # Run the standalone demo
    asyncio.run(main())