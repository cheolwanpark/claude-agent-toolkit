#!/usr/bin/env python3
# weather/main.py - Weather demo using Claude Agent Toolkit

import asyncio
import os
import sys

# Import from claude-agent-toolkit package
from claude_agent_toolkit import Agent, ConnectionError, ConfigurationError, ExecutionError

# Import our weather tool and prompts
from tool import WeatherTool
from prompt import WEATHER_SYSTEM_PROMPT, WELCOME_MESSAGE


async def run_weather_demo():
    """Run the weather demo with interactive examples."""
    print("\n" + "="*60)
    print("CLAUDE AGENT TOOLKIT WEATHER DEMO")
    print("="*60)
    print("\nThis demo showcases:")
    print("1. Real-time weather API integration (wttr.in)")
    print("2. Complex async tool operations")
    print("3. State management with favorites and history")
    print("4. Professional weather assistant behavior")
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return False
    
    try:
        # Start the weather tool
        weather_tool = WeatherTool()
        
        # Create agent with new pattern (system prompt + tools)
        agent = Agent(
            system_prompt=WEATHER_SYSTEM_PROMPT,
            tools=[weather_tool],
            model="haiku"  # Use fast Haiku model for weather queries
        )
        
        # NOTE: The old pattern still works:
        # agent = Agent()
        # agent.connect(weather_tool)
        
        print("\n📝 Starting Weather Agent Demo")
        print("-" * 40)
        
        # Demo 1: Current weather for a major city
        print(f"\n🌤️  Demo 1: Current Weather Conditions")
        try:
            response = await agent.run(
                "Please get the current weather conditions for Tokyo, Japan. "
                "Provide comprehensive information including temperature, humidity, wind, "
                "and practical advice about what to expect.",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 1]:")
            print(f"Response: {response[:1000]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 2: Multi-day forecast
        print(f"\n🌦️  Demo 2: Weather Forecast")
        try:
            response = await agent.run(
                "Now get a 3-day weather forecast for London, England. "
                "Help me understand what the weather will be like and what I should plan for.",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 2]:")
            print(f"Response: {response[:1000]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 3: Weather comparison
        print(f"\n🔄 Demo 3: Weather Comparison")
        try:
            response = await agent.run(
                "Compare the current weather between New York City and Los Angeles. "
                "Which city has better weather right now and why?",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 3]:")
            print(f"Response: {response[:1000]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 4: Travel weather planning
        print(f"\n✈️ Demo 4: Travel Weather Planning")
        try:
            response = await agent.run(
                "I'm planning to travel from Paris to Sydney next week. "
                "Can you help me understand what weather to expect and what to pack? "
                "Also, add both cities to my favorites list for future reference.",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 4]:")
            print(f"Response: {response[:1200]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Demo 5: State management - check favorites and history
        print(f"\n📊 Demo 5: State Management")
        try:
            response = await agent.run(
                "Show me my weather query history and favorite locations. "
                "What locations have I been checking the weather for?",
                verbose=True
            )
            
            print(f"\n[Agent Response - Demo 5]:")
            print(f"Response: {response[:800]}...")
        except (ConfigurationError, ConnectionError, ExecutionError) as e:
            print(f"❌ Error: {e}")
            return
        
        # Verify that weather queries were actually made by checking tool instance
        query_count = weather_tool.query_count
        favorites_count = len(weather_tool.favorite_locations)
        
        if query_count > 0:
            print(f"\n✅ SUCCESS: Weather tool was used {query_count} times")
            print(f"Favorites added: {favorites_count}")
            print(f"Last location queried: {weather_tool.last_location}")
            return True
        else:
            print(f"\n❌ FAILURE: Weather tool was not used")
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
        print(f"\n❌ Error during weather demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print("\n" + "="*60)
        print("WEATHER DEMO COMPLETED")
        print("="*60)


async def run_interactive_mode():
    """Run the weather assistant in interactive mode for user input."""
    print(WELCOME_MESSAGE)
    
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return
    
    try:
        # Start the weather tool
        weather_tool = WeatherTool(workers=2)
        
        # Create agent with system prompt for interactive mode
        agent = Agent(
            system_prompt=WEATHER_SYSTEM_PROMPT,
            tools=[weather_tool],
            model="haiku"  # Use fast Haiku model for interactive weather queries
        )
        
        print(f"\n🤖 Weather assistant is ready! Type 'quit' to exit.")
        print(f"💡 Try: 'What's the weather in Paris?' or 'Compare NYC and LA weather'")
        
        while True:
            user_input = input("\n📝 Your weather question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            try:
                response = await agent.run(f"User question: {user_input}", verbose=True)
                print(f"\n🤖 Weather Assistant: {response}")
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
    """Main entry point for the weather demo."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await run_interactive_mode()
    else:
        success = await run_weather_demo()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())