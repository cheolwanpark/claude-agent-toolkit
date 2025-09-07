#!/usr/bin/env python3
# subprocess/main.py - Test subprocess executor with SimpleTool

import asyncio
import os
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ExecutorType, ConnectionError, ConfigurationError, ExecutionError

# Import our simple tool
from tool import SimpleTool


async def run_subprocess_demo():
    """Run the subprocess executor demo."""
    print("\n" + "="*60)
    print("CLAUDE AGENT TOOLKIT SUBPROCESS EXECUTOR DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("1. SubprocessExecutor instead of Docker execution")
    print("2. Simple tool implementation (SimpleTool)")
    print("3. Direct subprocess execution without containers")
    print("4. Testing MCP tool integration with subprocess")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return False
    
    try:
        # Create the simple tool
        simple_tool = SimpleTool()
        
        # Create agent with subprocess executor preference
        agent = Agent(
            system_prompt="You are a helpful assistant that can echo messages. Use the available tools to respond to user requests.",
            tools=[simple_tool]
        )
        
        print(f"\n📝 Starting Subprocess Executor Demo")
        print("-" * 40)
        
        # Demo 1: Basic echo test
        print(f"\n🔄 Demo 1: Basic Echo Test")
        try:
            response = await agent.run(
                "Please use the echo tool to echo the message 'Hello from subprocess executor!'",
                executor=ExecutorType.SUBPROCESS,  # Key: Use subprocess instead of Docker
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
                "Please echo 'First message' and then echo 'Second message', then check the tool status.",
                executor=ExecutorType.SUBPROCESS,
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
                "Please get the history of echo calls and show me the current tool status.",
                executor=ExecutorType.SUBPROCESS,
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 3]:")
            print(f"Response: {response[:500]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error in Demo 3: {e}")
            return False
        
        # Verify that tools were actually called
        call_count = simple_tool.call_count
        message_history = len(simple_tool.messages)
        
        if call_count > 0:
            print(f"\n✅ SUCCESS: SimpleTool was called {call_count} times")
            print(f"✅ Message history contains {message_history} entries")
            print(f"✅ Subprocess executor is working correctly!")
            return True
        else:
            print(f"\n❌ FAILURE: SimpleTool was not called")
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
        # Create the simple tool
        simple_tool = SimpleTool()
        
        # Create agent
        agent = Agent(
            system_prompt="You are a helpful assistant with access to echo and status tools. Always use the tools when appropriate.",
            tools=[simple_tool]
        )
        
        print(f"\n🤖 Subprocess agent is ready! Type 'quit' to exit.")
        print(f"Available commands: echo messages, check status, view history")
        
        while True:
            user_input = input("\n📝 Your command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                response = await agent.run(
                    f"User request: {user_input}",
                    executor=ExecutorType.SUBPROCESS,
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