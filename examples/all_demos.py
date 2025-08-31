#!/usr/bin/env python3
# all_demos.py - Combined orchestrator for all verification demos

import asyncio
import os

# Import main functions from independent scripts
from examples.secret_example import main as secret_demo
from examples.hash_example import main as hash_demo
from examples.multi_tool_example import main as multi_tool_demo


async def main():
    """Run all verification demos in sequence."""
    print("\n" + "="*60)
    print("MCP TOOLS VERIFICATION DEMO - ALL EXAMPLES")
    print("="*60)
    print("\nThis orchestrates all verification demos:")
    print("1. Secret generator verification")
    print("2. Hash computer verification") 
    print("3. Multi-tool coordination verification")
    print("\nNote: Each demo is independent and can be run separately:")
    print("- python examples/secret_example.py")
    print("- python examples/hash_example.py")
    print("- python examples/multi_tool_example.py")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        return
    
    try:
        # Run all verification demos
        test_results = []
        
        print("\n" + "="*60)
        print("RUNNING DEMO 1 OF 3")
        print("="*60)
        test1_passed = await secret_demo()
        test_results.append(("Secret Generator", test1_passed))
        
        print("\n" + "="*60)
        print("RUNNING DEMO 2 OF 3")
        print("="*60)
        test2_passed = await hash_demo()
        test_results.append(("Hash Computer", test2_passed))
        
        print("\n" + "="*60)
        print("RUNNING DEMO 3 OF 3")
        print("="*60)
        test3_passed = await multi_tool_demo()
        test_results.append(("Multi-Tool", test3_passed))
        
        # Summary
        print("\n" + "="*60)
        print("OVERALL VERIFICATION SUMMARY")
        print("="*60)
        
        for test_name, passed in test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        all_passed = all(passed for _, passed in test_results)
        if all_passed:
            print("\n🎉 All verification tests passed! The agent is calling tools correctly.")
        else:
            print("\n⚠️  Some tests failed. The agent may not be calling tools properly.")
        
    except RuntimeError as e:
        if "Cannot connect to Docker" in str(e):
            print(f"\n{e}")
            print("\n💡 After starting Docker, run this demo again.")
        else:
            raise
    except Exception as e:
        print(f"\n❌ Error during demo orchestration: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run all demos
    asyncio.run(main())