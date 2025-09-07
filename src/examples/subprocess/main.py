#!/usr/bin/env python3
# subprocess/main.py - Test subprocess executor with SimpleTool

import asyncio
import os
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ExecutorType, ConnectionError, ConfigurationError, ExecutionError

# Import our message reflector tool
from tool import MessageReflector


async def run_subprocess_demo():
    """Run the subprocess executor demo."""
    print("\n" + "="*60)
    print("CLAUDE AGENT TOOLKIT SUBPROCESS EXECUTOR DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("1. SubprocessExecutor instead of Docker execution")
    print("2. MessageReflector tool implementation with distinct naming")
    print("3. Direct subprocess execution without containers")
    print("4. Testing MCP tool integration with subprocess")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return False
    
    try:
        # Create the message reflector tool
        reflector_tool = MessageReflector()
        
        # Create agent with subprocess executor preference
        agent = Agent(
            system_prompt="""You are a helpful assistant with MessageReflector tool functions. Use ONLY these specific tool IDs:
- mcp__messagereflector__reflect_message: Reflect messages back with metadata
- mcp__messagereflector__get_reflection_history: Get history of reflections  
- mcp__messagereflector__get_tool_status: Get current tool status
DO NOT use bash echo or any other tools. Always use the exact mcp__messagereflector__ tool IDs listed above.""",
            tools=[reflector_tool],
            executor=ExecutorType.SUBPROCESS  # Use subprocess instead of Docker
        )
        
        print(f"\n📝 Starting Subprocess Executor Demo")
        print("-" * 40)
        
        # Demo 1: Basic reflection test
        print(f"\n🔄 Demo 1: Basic Message Reflection Test")
        try:
            response = await agent.run(
                "Please use the exact tool ID mcp__messagereflector__reflect_message to reflect back the message 'Hello from subprocess executor!'",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 1]:")
            print(f"Response: {response[:500]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error in Demo 1: {e}")
            return False
        
        # Demo 2: Multiple tool calls
        print(f"\n🔄 Demo 2: Multiple Tool Interactions")
        try:
            response = await agent.run(
                "Please use mcp__messagereflector__reflect_message to reflect 'First message' and then reflect 'Second message', then use mcp__messagereflector__get_tool_status to check the tool status.",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 2]:")
            print(f"Response: {response[:500]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error in Demo 2: {e}")
            return False
        
        # Demo 3: History check
        print(f"\n🔄 Demo 3: History and Status Check")
        try:
            response = await agent.run(
                "Please use mcp__messagereflector__get_reflection_history to get the reflection history and mcp__messagereflector__get_tool_status to show me the current tool status.",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 3]:")
            print(f"Response: {response[:500]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error in Demo 3: {e}")
            return False
        
        # Verify that tools were actually called
        call_count = reflector_tool.call_count
        message_history = len(reflector_tool.messages)
        
        if call_count > 0:
            print(f"\n✅ SUCCESS: MessageReflector tool was called {call_count} times")
            print(f"✅ Message history contains {message_history} entries")
            print(f"✅ Subprocess executor is working correctly!")
            return True
        else:
            print(f"\n❌ FAILURE: MessageReflector tool was not called")
            return False
            
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 This is expected - subprocess executor doesn't need Docker!")
        print("💡 Check that the subprocess executor is handling this correctly.")
        return False
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Check your OAuth token and tool configuration.")
        return False
    except ExecutionError as e:
        print(f"\n❌ Execution Error: {e}")
        print("\n💡 The subprocess execution failed. Check the error details above.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during subprocess demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "="*60)
        print("SUBPROCESS EXECUTOR DEMO COMPLETED")
        print("="*60)


async def run_interactive_mode():
    """Run the subprocess executor in interactive mode."""
    print("\n" + "="*50)
    print("INTERACTIVE SUBPROCESS EXECUTOR MODE")
    print("="*50)
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return
    
    try:
        # Create the message reflector tool
        reflector_tool = MessageReflector()
        
        # Create agent
        agent = Agent(
            system_prompt="""You are a helpful assistant with MessageReflector tool functions. Use ONLY these exact tool IDs:
- mcp__messagereflector__reflect_message: Reflect messages back
- mcp__messagereflector__get_reflection_history: Get reflection history
- mcp__messagereflector__get_tool_status: Get tool status
DO NOT use bash echo or any other MCP tools. Always use the exact mcp__messagereflector__ tool IDs.""",
            tools=[reflector_tool],
            executor=ExecutorType.SUBPROCESS  # Use subprocess instead of Docker
        )
        
        print(f"\n🤖 Subprocess agent is ready! Type 'quit' to exit.")
        print(f"Available commands: reflect messages, check status, view reflection history")
        
        while True:
            user_input = input("\n📝 Your command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                response = await agent.run(
                    f"User request: {user_input}",
                    verbose=True
                )
                print(f"\n🤖 Assistant: {response}")
            except (ConfigurationError, ConnectionError, ExecutionError) as e:
                print(f"\n❌ Error: {e}")
    
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 Check subprocess executor setup.")
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Check your OAuth token and tool configuration.")
    except ExecutionError as e:
        print(f"\n❌ Execution Error: {e}")
        print("\n💡 The subprocess execution failed. Try rephrasing your request.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error in interactive mode: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point for the subprocess executor demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await run_interactive_mode()
    else:
        success = await run_subprocess_demo()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())