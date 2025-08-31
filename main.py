#!/usr/bin/env python3
# main.py - Demo application showing tools and agents

import asyncio
import secrets
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional
from tool import BaseTool, tool
from agent import Agent


# ============= Tool 1: Secret Generator Tool =============
class SecretGeneratorTool(BaseTool):
    """A tool that generates and stores secret values."""
    
    def __init__(self):
        super().__init__()
        self.state = {
            "current_secret": None,
            "secret_history": [],
            "generation_count": 0
        }
    
    @tool(
        description="Generate a new random secret token",
        cpu_bound=False
    )
    async def generate_secret(self) -> Dict[str, Any]:
        """Generate a new secret token."""
        # Generate unpredictable secret
        new_secret = secrets.token_hex(16)
        timestamp = datetime.now().isoformat()
        
        # Store in state
        self.state["current_secret"] = new_secret
        self.state["generation_count"] += 1
        self.state["secret_history"].append({
            "secret": new_secret,
            "timestamp": timestamp,
            "index": self.state["generation_count"]
        })
        
        print(f"\n🔐 [Tool] Generated secret #{self.state['generation_count']}: {new_secret}\n")
        
        return {
            "secret": new_secret,
            "timestamp": timestamp,
            "message": f"Secret #{self.state['generation_count']} generated successfully"
        }
    
    @tool(
        description="Get the current secret token",
        cpu_bound=False
    )
    async def get_current_secret(self) -> Dict[str, Any]:
        """Retrieve the current secret."""
        if self.state["current_secret"] is None:
            return {
                "secret": None,
                "message": "No secret has been generated yet"
            }
        
        return {
            "secret": self.state["current_secret"],
            "generation_count": self.state["generation_count"]
        }
    
    @tool(
        description="Verify if a given token matches the current secret",
        cpu_bound=False
    )
    async def verify_secret(self, token: str) -> Dict[str, Any]:
        """Verify if a token matches the current secret."""
        is_valid = token == self.state["current_secret"]
        
        return {
            "valid": is_valid,
            "provided_token": token,
            "message": "Token is valid!" if is_valid else "Token does not match"
        }


# ============= Tool 2: Hash Computer Tool =============
class HashComputerTool(BaseTool):
    """A tool that computes hashes with timestamps."""
    
    def __init__(self):
        super().__init__()
        self.state = {
            "computed_hashes": [],
            "last_hash": None,
            "computation_count": 0
        }
    
    @tool(
        description="Compute SHA256 hash of input with current timestamp",
        cpu_bound=True,
        snapshot=[]
    )
    def compute_hash(self, input_text: str) -> Dict[str, Any]:
        """Compute hash of input combined with timestamp."""
        # Add timestamp to make it unpredictable
        timestamp = str(time.time())
        combined = f"{input_text}:{timestamp}"
        
        # Compute hash
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        # Update state
        self.state["computation_count"] += 1
        self.state["last_hash"] = hash_value
        self.state["computed_hashes"].append({
            "input": input_text,
            "timestamp": timestamp,
            "hash": hash_value,
            "index": self.state["computation_count"]
        })
        
        print(f"\n#️⃣ [Tool] Computed hash #{self.state['computation_count']}: {hash_value[:16]}...\n")
        
        return {
            "hash": hash_value,
            "input": input_text,
            "timestamp": timestamp,
            "message": f"Hash #{self.state['computation_count']} computed"
        }
    
    @tool(
        description="Get the last computed hash",
        cpu_bound=False
    )
    async def get_last_hash(self) -> Dict[str, Any]:
        """Get the most recently computed hash."""
        if self.state["last_hash"] is None:
            return {
                "hash": None,
                "message": "No hash has been computed yet"
            }
        
        last_entry = self.state["computed_hashes"][-1] if self.state["computed_hashes"] else None
        
        return {
            "hash": self.state["last_hash"],
            "computation_count": self.state["computation_count"],
            "details": last_entry
        }
    
    @tool(
        description="Combine two strings and compute their hash",
        cpu_bound=True,
        snapshot=[]
    )
    def combine_and_hash(self, first: str, second: str) -> Dict[str, Any]:
        """Combine two strings and compute hash."""
        timestamp = str(time.time())
        combined = f"{first}|{second}|{timestamp}"
        hash_value = hashlib.sha256(combined.encode()).hexdigest()
        
        self.state["computation_count"] += 1
        self.state["last_hash"] = hash_value
        self.state["computed_hashes"].append({
            "first": first,
            "second": second,
            "timestamp": timestamp,
            "hash": hash_value,
            "index": self.state["computation_count"]
        })
        
        print(f"\n#️⃣ [Tool] Combined hash #{self.state['computation_count']}: {hash_value[:16]}...\n")
        
        return {
            "hash": hash_value,
            "first": first,
            "second": second,
            "timestamp": timestamp
        }


# ============= Demo Functions =============
async def demo_secret_verification():
    """Demo: Verify that agent actually calls the secret generator tool."""
    print("\n" + "="*60)
    print("DEMO 1: Secret Generator Verification")
    print("="*60)
    
    # Start the secret generator tool
    secret_tool = SecretGeneratorTool().run(workers=2)
    
    # Create agent and connect to tool
    agent = Agent()
    agent.connect(secret_tool)
    
    print("\n📝 Test 1: Generate and retrieve a secret")
    print("-" * 40)
    
    # Ask agent to generate and tell us the secret
    result = await agent.run(
        "Please generate a new secret token using the generate_secret tool, "
        "then tell me exactly what the secret token is."
    )
    
    print(f"\n[Agent Response]:")
    print(f"Success: {result.get('success')}")
    response = result.get('response', '')
    print(f"Response: {response[:500]}...")
    
    # Verify the secret was actually generated by checking tool state
    actual_secret = secret_tool.state.get("current_secret")
    generation_count = secret_tool.state.get("generation_count", 0)
    
    print(f"\n✅ Verification:")
    print(f"  - Tool was called: {generation_count > 0}")
    print(f"  - Actual secret from tool: {actual_secret}")
    print(f"  - Secret appears in response: {actual_secret in response if actual_secret else False}")
    
    # Test 2: Verify a specific token
    print("\n📝 Test 2: Verify a specific token")
    print("-" * 40)
    
    test_token = "test123"
    result = await agent.run(
        f"Please verify if the token '{test_token}' matches the current secret "
        "using the verify_secret tool. Tell me if it's valid or not."
    )
    
    print(f"\n[Agent Response]:")
    print(f"Response: {result.get('response', '')[:300]}...")
    
    return generation_count > 0 and actual_secret is not None


async def demo_hash_verification():
    """Demo: Verify that agent actually calls the hash computer tool."""
    print("\n" + "="*60)
    print("DEMO 2: Hash Computer Verification")
    print("="*60)
    
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
    
    return computation_count > 0 and last_hash is not None


async def demo_multi_tool_verification():
    """Demo: Verify agent can use multiple tools and actually calls them."""
    print("\n" + "="*60)
    print("DEMO 3: Multi-Tool Verification")
    print("="*60)
    
    # Start both tools
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
    
    return secret_generated and hash_computed


async def main():
    """Main demo application."""
    print("\n" + "="*60)
    print("MCP TOOLS VERIFICATION DEMO")
    print("="*60)
    print("\nThis demo verifies that agents actually call tools by:")
    print("1. Using tools that generate unpredictable results (secrets, hashes)")
    print("2. Checking tool state to confirm they were called")
    print("3. Verifying the unpredictable values appear in agent responses")
    
    # Check for OAuth token
    import os
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        return
    
    try:
        # Run verification demos
        test_results = []
        
        test1_passed = await demo_secret_verification()
        test_results.append(("Secret Generator", test1_passed))
        
        test2_passed = await demo_hash_verification()
        test_results.append(("Hash Computer", test2_passed))
        
        test3_passed = await demo_multi_tool_verification()
        test_results.append(("Multi-Tool", test3_passed))
        
        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
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
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())