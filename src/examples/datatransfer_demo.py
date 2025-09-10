#!/usr/bin/env python3
# datatransfer_demo.py - Demonstration of DataTransferTool with Pydantic models

import asyncio
import os
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr

# Import from claude-agent-toolkit
from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import DataTransferTool


# Define example Pydantic models for demonstration
class UserProfile(BaseModel):
    """User profile data model."""
    name: str = Field(..., description="Full name of the user")
    age: int = Field(..., ge=0, le=150, description="Age in years")
    email: str = Field(..., description="Email address")
    bio: Optional[str] = Field(None, description="User biography")
    interests: List[str] = Field(default_factory=list, description="List of user interests")


class ProductInfo(BaseModel):
    """Product information model."""
    name: str = Field(..., description="Product name")
    price: float = Field(..., gt=0, description="Price in USD")
    description: str = Field(..., description="Product description")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")


async def demonstrate_user_profile_transfer():
    """Demonstrate transferring user profile data."""
    print("\n" + "="*60)
    print("USER PROFILE TRANSFER DEMO")
    print("="*60)
    
    # Create a DataTransferTool for UserProfile with distinct name
    user_tool = DataTransferTool.create(UserProfile, "UserProfileTool")
    
    # Create agent with the tool
    agent = Agent(
        system_prompt="You are a data assistant that helps transfer and validate user data.",
        tools=[user_tool]
    )
    
    print("\n📝 Transferring user profile data...")
    
    # Ask Claude to transfer user data
    response = await agent.run(
        "Please transfer this user data: "
        "name='Alice Johnson', age=28, email='alice@example.com', "
        "bio='Software engineer who loves hiking', "
        "interests=['programming', 'hiking', 'photography']"
    )
    
    print(f"\n🤖 Agent Response:")
    print(response)
    
    # Retrieve the transferred data from the host side
    user_data = user_tool.get()
    if user_data:
        print(f"\n✅ Successfully received user profile:")
        print(f"   Name: {user_data.name}")
        print(f"   Age: {user_data.age}")
        print(f"   Email: {user_data.email}")
        print(f"   Bio: {user_data.bio}")
        print(f"   Interests: {', '.join(user_data.interests)}")
        
        # Demonstrate JSON serialization
        print(f"\n📄 JSON representation:")
        print(user_data.model_dump_json(indent=2))
    else:
        print("\n❌ No data was transferred")


async def demonstrate_product_transfer():
    """Demonstrate transferring product data."""
    print("\n" + "="*60)
    print("PRODUCT INFORMATION TRANSFER DEMO")
    print("="*60)
    
    # Create a DataTransferTool for ProductInfo with distinct name
    product_tool = DataTransferTool.create(ProductInfo, "ProductInfoTool")
    
    # Create agent with the tool
    agent = Agent(
        system_prompt="You are a product data assistant that validates and transfers product information.",
        tools=[product_tool]
    )
    
    print("\n📝 Transferring product information...")
    
    # Ask Claude to transfer product data
    response = await agent.run(
        "Transfer this product data: "
        "name='Wireless Headphones', price=199.99, "
        "description='High-quality wireless headphones with noise cancellation', "
        "category='Electronics', in_stock=true"
    )
    
    print(f"\n🤖 Agent Response:")
    print(response)
    
    # Retrieve the transferred data
    product_data = product_tool.get()
    if product_data:
        print(f"\n✅ Successfully received product information:")
        print(f"   Product: {product_data.name}")
        print(f"   Price: ${product_data.price}")
        print(f"   Category: {product_data.category}")
        print(f"   Description: {product_data.description}")
        print(f"   In Stock: {'Yes' if product_data.in_stock else 'No'}")
        
        # Show schema information
        print(f"\n📋 Product Schema:")
        schema = product_tool.get_schema()
        for field_name, field_info in schema['properties'].items():
            field_type = field_info.get('type', 'unknown')
            field_desc = field_info.get('description', 'No description')
            print(f"   {field_name}: {field_type} - {field_desc}")
    else:
        print("\n❌ No product data was transferred")


async def demonstrate_validation_error():
    """Demonstrate validation error handling."""
    print("\n" + "="*60)
    print("VALIDATION ERROR DEMO")
    print("="*60)
    
    # Create a DataTransferTool for UserProfile with distinct name
    user_tool = DataTransferTool.create(UserProfile, "UserValidationTool")
    
    # Create agent with the tool
    agent = Agent(
        system_prompt="You are a data assistant. Transfer data exactly as provided.",
        tools=[user_tool]
    )
    
    print("\n📝 Attempting to transfer invalid user data...")
    
    # Ask Claude to transfer invalid data (negative age)
    response = await agent.run(
        "Transfer this user data with invalid values: "
        "name='Bob Smith', age=-5, email='not-an-email', bio=null"
    )
    
    print(f"\n🤖 Agent Response (should show validation errors):")
    print(response)
    
    # Check if any data was transferred
    user_data = user_tool.get()
    if user_data:
        print(f"\n✅ Data was transferred despite errors: {user_data}")
    else:
        print("\n❌ No data was transferred due to validation errors (expected)")


async def main():
    """Main demo function."""
    # Check for OAuth token
    if not os.environ.get('CLAUDE_CODE_OAUTH_TOKEN'):
        print("\n⚠️  WARNING: CLAUDE_CODE_OAUTH_TOKEN not set!")
        print("Please set your OAuth token: export CLAUDE_CODE_OAUTH_TOKEN='your-token'")
        print("Get your token from: https://claude.ai/code")
        return
    
    print("\n🚀 DATATRANSFER TOOL DEMONSTRATION")
    print("This demo shows how to use DataTransferTool with Pydantic models")
    
    try:
        # Run all demonstrations
        await demonstrate_user_profile_transfer()
        await demonstrate_product_transfer()
        await demonstrate_validation_error()
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey features demonstrated:")
        print("• Generic tool that works with any Pydantic BaseModel")
        print("• Automatic schema inclusion in tool descriptions")
        print("• Type-safe data validation and transfer")
        print("• Clear error messages for validation failures")
        print("• Simple transfer/get interface for host applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())