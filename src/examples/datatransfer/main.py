#!/usr/bin/env python3
# main.py - DataTransfer Tool Demonstration Entry Point

import asyncio
import os
import time

import single_model
import multiple_models
import complex


async def main():
    """Run all DataTransfer tool demonstrations."""
    print("🚀 DATATRANSFER TOOL COMPREHENSIVE DEMONSTRATIONS")
    print("=" * 70)
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return
    
    start_time = time.time()
    
    try:
        print("\nRunning comprehensive DataTransfer demonstrations...")
        print("This shows DataTransferTool capabilities across various scenarios.\n")
        
        # Run all demo categories
        await single_model.run_all()
        await multiple_models.run_all()
        await complex.run_all()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✅ ALL DATATRANSFER DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        print(f"\nExecution time: {elapsed:.1f} seconds")
        
        print("\nKey features demonstrated:")
        print("• Generic tool that works with any Pydantic BaseModel")
        print("• Dynamic class creation for distinct tool identities")  
        print("• Automatic schema inclusion in tool descriptions")
        print("• Type-safe data validation and transfer")
        print("• Clear error messages for validation failures")
        print("• Nested model support for complex data structures")
        print("• Multiple tools in single agent scenarios")
        print("• Data transformation between different models")
        print("• Field constraints and validation handling")
        print("• Simple transfer/get interface for host applications")
        
        print(f"\n🎉 DataTransferTool is production-ready!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)