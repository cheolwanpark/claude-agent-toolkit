#!/usr/bin/env python3
# single_model.py - Single model transfer demonstrations

from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import DataTransferTool
from models import UserProfile, ProductInfo, Configuration


async def demo_user_profile():
    """Demonstrate transferring user profile data."""
    print("\n" + "="*50)
    print("User Profile Transfer")
    print("="*50)
    
    # Create tool and agent
    user_tool = DataTransferTool.create(UserProfile, "UserProfileTool")
    agent = Agent(
        system_prompt="You are a data assistant for user profile transfers.",
        tools=[user_tool]
    )
    
    print("\n📝 Transferring user profile...")
    
    # Transfer user data
    response = await agent.run(
        "Transfer this user data: "
        "name='Alice Johnson', age=28, email='alice@example.com', "
        "bio='Software engineer', interests=['programming', 'hiking']"
    )
    
    print(f"\n🤖 Response: {response[:200]}...")
    
    # Retrieve and display data
    user_data = user_tool.get()
    if user_data:
        print(f"\n✅ Retrieved: {user_data.name}, {user_data.age}, {user_data.email}")
    else:
        print("\n❌ No data retrieved")


async def demo_product_info():
    """Demonstrate transferring product information."""
    print("\n" + "="*50)
    print("Product Info Transfer")
    print("="*50)
    
    # Create tool and agent
    product_tool = DataTransferTool.create(ProductInfo, "ProductTool")
    agent = Agent(
        system_prompt="You are a product data assistant.",
        tools=[product_tool]
    )
    
    print("\n📝 Transferring product data...")
    
    # Transfer product data
    response = await agent.run(
        "Transfer product: name='Wireless Mouse', price=29.99, "
        "description='Ergonomic wireless mouse', category='Electronics', in_stock=true"
    )
    
    print(f"\n🤖 Response: {response[:200]}...")
    
    # Retrieve and display data
    product_data = product_tool.get()
    if product_data:
        print(f"\n✅ Retrieved: {product_data.name} - ${product_data.price}")
    else:
        print("\n❌ No data retrieved")


async def demo_configuration():
    """Demonstrate transferring configuration data with various field types."""
    print("\n" + "="*50)
    print("Configuration Transfer")
    print("="*50)
    
    # Create tool and agent
    config_tool = DataTransferTool.create(Configuration, "ConfigTool")
    agent = Agent(
        system_prompt="You are a configuration data assistant.",
        tools=[config_tool]
    )
    
    print("\n📝 Transferring configuration...")
    
    # Transfer configuration data
    response = await agent.run(
        "Transfer config: name='api_config', enabled=true, timeout_seconds=120, "
        "retry_count=5, allowed_hosts=['api.example.com', 'backup.example.com']"
    )
    
    print(f"\n🤖 Response: {response[:200]}...")
    
    # Retrieve and display data
    config_data = config_tool.get()
    if config_data:
        print(f"\n✅ Retrieved: {config_data.name}, timeout={config_data.timeout_seconds}s")
    else:
        print("\n❌ No data retrieved")


async def run_all():
    """Run all single model demonstrations."""
    print("\n" + "="*60)
    print("SINGLE MODEL TRANSFER DEMOS")
    print("="*60)
    
    await demo_user_profile()
    await demo_product_info()
    await demo_configuration()
    
    print("\n✅ Single model demos completed")