#!/usr/bin/env python3
# multiple_models.py - Multiple model transfer demonstrations

from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import DataTransferTool
from models import UserProfile, ProductInfo, CustomerInfo


async def demo_sequential_transfers():
    """Demonstrate transferring different models in sequence."""
    print("\n" + "="*50)
    print("Sequential Model Transfers")
    print("="*50)
    
    # Create separate agents for each model type
    user_tool = DataTransferTool.create(UserProfile, "UserTool")
    product_tool = DataTransferTool.create(ProductInfo, "ProductTool")
    
    user_agent = Agent(
        system_prompt="You handle user data transfers only.",
        tools=[user_tool]
    )
    
    product_agent = Agent(
        system_prompt="You handle product data transfers only.",
        tools=[product_tool]
    )
    
    print("\n📝 Step 1: Transfer user data...")
    user_response = await user_agent.run(
        "Transfer user: name='Bob Wilson', age=35, email='bob@company.com'"
    )
    print(f"✅ User: {user_response[:100]}...")
    
    print("\n📝 Step 2: Transfer product data...")
    product_response = await product_agent.run(
        "Transfer product: name='Laptop', price=899.99, category='Electronics'"
    )
    print(f"✅ Product: {product_response[:100]}...")
    
    # Retrieve both
    user_data = user_tool.get()
    product_data = product_tool.get()
    
    if user_data and product_data:
        print(f"\n✅ Got both: {user_data.name} and {product_data.name}")
    else:
        print("\n❌ Missing data")


async def demo_multiple_tools_one_agent():
    """Demonstrate one agent with multiple transfer tools."""
    print("\n" + "="*50)
    print("Multiple Tools - One Agent")
    print("="*50)
    
    # Create multiple tools
    user_tool = DataTransferTool.create(UserProfile, "UserDataTool")
    customer_tool = DataTransferTool.create(CustomerInfo, "CustomerDataTool")
    
    # Single agent with multiple tools
    agent = Agent(
        system_prompt="You can transfer both user profiles and customer info. "
                     "Use the appropriate tool for each data type.",
        tools=[user_tool, customer_tool]
    )
    
    print("\n📝 Transfer user profile...")
    user_response = await agent.run(
        "Transfer user profile: name='Carol Smith', age=30, email='carol@email.com', "
        "bio='Marketing manager', interests=['travel', 'photography']"
    )
    print(f"📋 User result: {user_response[:150]}...")
    
    print("\n📝 Transfer customer info...")
    customer_response = await agent.run(
        "Transfer customer info: customer_id='CUST001', name='David Brown', "
        "email='david@company.com', phone='555-0123'"
    )
    print(f"📋 Customer result: {customer_response[:150]}...")
    
    # Check both transfers
    user_data = user_tool.get()
    customer_data = customer_tool.get()
    
    if user_data and customer_data:
        print(f"\n✅ Retrieved both: User '{user_data.name}' and Customer '{customer_data.name}'")
    else:
        print("\n❌ Some data missing")


async def demo_data_transformation():
    """Demonstrate transforming data between different models."""
    print("\n" + "="*50)
    print("Data Transformation")
    print("="*50)
    
    # Step 1: Collect user data
    user_tool = DataTransferTool.create(UserProfile, "SourceUserTool")
    user_agent = Agent(
        system_prompt="Collect user profile data.",
        tools=[user_tool]
    )
    
    print("\n📝 Step 1: Collect user profile...")
    await user_agent.run(
        "Transfer: name='Emma Davis', age=27, email='emma@startup.com', "
        "bio='Product designer'"
    )
    
    user_data = user_tool.get()
    if not user_data:
        print("❌ Failed to get user data")
        return
    
    # Step 2: Transform to customer info
    customer_tool = DataTransferTool.create(CustomerInfo, "TransformedCustomerTool")
    transform_agent = Agent(
        system_prompt="Transform user data to customer format.",
        tools=[customer_tool]
    )
    
    print("\n📝 Step 2: Transform to customer format...")
    await transform_agent.run(
        f"Transform this user data to customer format: "
        f"Use name='{user_data.name}', email='{user_data.email}', "
        f"generate customer_id='CUST_{user_data.name.replace(' ', '').upper()}'"
    )
    
    customer_data = customer_tool.get()
    if customer_data:
        print(f"\n✅ Transformed: {user_data.name} → Customer ID: {customer_data.customer_id}")
    else:
        print("\n❌ Transformation failed")


async def run_all():
    """Run all multiple model demonstrations."""
    print("\n" + "="*60)
    print("MULTIPLE MODEL TRANSFER DEMOS")
    print("="*60)
    
    await demo_sequential_transfers()
    await demo_multiple_tools_one_agent()
    await demo_data_transformation()
    
    print("\n✅ Multiple model demos completed")