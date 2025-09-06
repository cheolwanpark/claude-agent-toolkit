#!/usr/bin/env python3
# calculator/main.py - Calculator demo using Claude Agent Toolkit

import asyncio
import os
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ConnectionError, ConfigurationError, ExecutionError

# Import our calculator tool and prompts
from tool import CalculatorTool
from prompt import CALCULATOR_SYSTEM_PROMPT, WELCOME_MESSAGE


async def run_calculator_demo():
    """Run the calculator demo with interactive examples."""
    print("\n" + "="*60)
    print("CLAUDE AGENT TOOLKIT CALCULATOR DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("1. Custom tool implementation (CalculatorTool)")
    print("2. State management across operations")
    print("3. Professional agent prompting")
    print("4. Multi-step mathematical problem solving")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return False
    
    try:
        # Start the calculator tool
        calculator_tool = CalculatorTool()
        
        # Create agent with new pattern (system prompt + tools)
        agent = Agent(
            system_prompt=CALCULATOR_SYSTEM_PROMPT,
            tools=[calculator_tool]
        )
        
        # NOTE: The old pattern still works:
        # agent = Agent()
        # agent.connect(calculator_tool)
        
        print("\n📝 Starting Calculator Agent Demo")
        print("-" * 40)
        
        # Demo 1: Basic arithmetic
        print(f"\n🧮 Demo 1: Basic Arithmetic Operations")
        try:
            response = await agent.run(
                "Please calculate: (25 + 75) × 3 - 50. "
                "Break this down step by step and show your work."
            )
            
            print(f"\n[Agent Response - Demo 1]:")
            print(f"Response: {response[:800]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 2: Advanced operations
        print(f"\n🧮 Demo 2: Advanced Mathematical Operations")
        try:
            response = await agent.run(
                "Now calculate the square root of the last result, then raise it to the power of 2.5. "
                "What do you notice about these operations?"
            )
            
            print(f"\n[Agent Response - Demo 2]:")
            print(f"Response: {response[:800]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 3: History and state management
        print(f"\n🧮 Demo 3: History and State Management")
        try:
            response = await agent.run(
                "Please show me the calculation history from our session, "
                "then explain what the last result represents mathematically."
            )
            
            print(f"\n[Agent Response - Demo 3]:")
            print(f"Response: {response[:800]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 4: Complex word problem
        print(f"\n🧮 Demo 4: Complex Problem Solving")
        response = await agent.run(
            "Solve this problem step by step: "
            "A rectangular garden is 15 meters long and 8 meters wide. "
            "If I want to put a fence around it that costs $12 per meter, "
            "and I also want to cover the entire area with grass seed that costs $3 per square meter, "
            "what will be the total cost?"
        )
        
        print(f"\n[Agent Response - Demo 4]:")
        print(f"Response: {response[:1000]}...")
        
        # Verify that calculations were actually performed by checking tool instance
        operation_count = calculator_tool.operation_count
        
        if operation_count > 0:
            print(f"\n✅ SUCCESS: Calculator tool was used {operation_count} times")
            print(f"Last result: {calculator_tool.last_result}")
            return True
        else:
            print(f"\n❌ FAILURE: Calculator tool was not used")
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
        print(f"\n❌ Unexpected error during calculator demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "="*60)
        print("CALCULATOR DEMO COMPLETED")
        print("="*60)


async def run_interactive_mode():
    """Run the calculator in interactive mode for user input."""
    print(WELCOME_MESSAGE)
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return
    
    try:
        # Start the calculator tool
        calculator_tool = CalculatorTool(workers=2)
        
        # Create agent with system prompt for interactive mode
        agent = Agent(
            system_prompt=CALCULATOR_SYSTEM_PROMPT,
            tools=[calculator_tool]
        )
        
        print(f"\n🤖 Calculator agent is ready! Type 'quit' to exit.")
        
        while True:
            user_input = input("\n📝 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                response = await agent.run(f"User question: {user_input}")
                print(f"\n🤖 Assistant: {response}")
            except (ConfigurationError, ConnectionError, ExecutionError) as e:
                print(f"\n❌ Error: {e}")
    
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        if "Docker" in str(e):
            print("\n💡 Please start Docker Desktop and try again.")
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 Check your OAuth token and tool configuration.")
    except ExecutionError as e:
        print(f"\n❌ Execution Error: {e}")
        print("\n💡 The agent execution failed. Try rephrasing your question.")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error in interactive mode: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point for the calculator demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await run_interactive_mode()
    else:
        success = await run_calculator_demo()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())