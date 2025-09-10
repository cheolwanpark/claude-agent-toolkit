#!/usr/bin/env python3
# complex.py - Complex scenario demonstrations

from claude_agent_toolkit import Agent
from claude_agent_toolkit.tools import DataTransferTool
from models import UserProfile, Order, Configuration, Team, Company


async def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("\n" + "="*50)
    print("Validation Error Handling")
    print("="*50)
    
    user_tool = DataTransferTool.create(UserProfile, "ValidationUserTool")
    agent = Agent(
        system_prompt="Transfer data exactly as provided, even if invalid.",
        tools=[user_tool]
    )
    
    print("\n📝 Attempting invalid data transfer...")
    
    # Try to transfer invalid data
    response = await agent.run(
        "Transfer user with invalid data: "
        "name='Invalid User', age=-10, email='not-an-email'"
    )
    
    print(f"\n🤖 Response: {response[:300]}...")
    
    # Check if data was transferred
    user_data = user_tool.get()
    if user_data:
        print(f"\n❌ Unexpected: Data was transferred: {user_data}")
    else:
        print(f"\n✅ Expected: No data transferred due to validation errors")


async def demo_nested_models():
    """Demonstrate transferring nested model structures."""
    print("\n" + "="*50)
    print("Nested Model Transfer")
    print("="*50)
    
    order_tool = DataTransferTool.create(Order, "OrderTool")
    agent = Agent(
        system_prompt="You transfer complex order data with nested customer and items.",
        tools=[order_tool]
    )
    
    print("\n📝 Transferring complex order with nested data...")
    
    response = await agent.run(
        "Transfer order: order_id='ORD001', "
        "customer: {customer_id='CUST001', name='John Doe', email='john@email.com'}, "
        "items: [{product_name='Laptop', quantity=1, unit_price=999.99}, "
        "{product_name='Mouse', quantity=2, unit_price=25.00}], "
        "total_amount=1049.99, status='processing'"
    )
    
    print(f"\n🤖 Response: {response[:300]}...")
    
    order_data = order_tool.get()
    if order_data:
        print(f"\n✅ Retrieved order: {order_data.order_id}")
        print(f"   Customer: {order_data.customer.name}")
        print(f"   Items: {len(order_data.items)} items")
        print(f"   Total: ${order_data.total_amount}")
    else:
        print("\n❌ No order data retrieved")


async def demo_field_constraints():
    """Demonstrate field constraints and validation."""
    print("\n" + "="*50)
    print("Field Constraints Demo")
    print("="*50)
    
    config_tool = DataTransferTool.create(Configuration, "ConstraintConfigTool")
    agent = Agent(
        system_prompt="Transfer configuration data following all field constraints.",
        tools=[config_tool]
    )
    
    print("\n📝 Valid configuration...")
    valid_response = await agent.run(
        "Transfer valid config: name='prod_config', enabled=true, "
        "timeout_seconds=300, retry_count=3, "
        "allowed_hosts=['prod.api.com', 'backup.api.com']"
    )
    print(f"✅ Valid: {valid_response[:200]}...")
    
    config_data = config_tool.get()
    if config_data:
        print(f"✅ Config transferred: {config_data.name}")
    
    # Clear for next test
    config_tool.clear()
    
    print("\n📝 Invalid configuration (timeout too high)...")
    invalid_response = await agent.run(
        "Transfer invalid config: name='bad_config', timeout_seconds=5000"
    )
    print(f"❌ Invalid: {invalid_response[:200]}...")
    
    config_data_invalid = config_tool.get()
    if config_data_invalid:
        print("❌ Unexpected: Invalid config was transferred")
    else:
        print("✅ Expected: Invalid config rejected")


async def demo_large_data():
    """Demonstrate handling larger data structures."""
    print("\n" + "="*50)
    print("Large Data Transfer")
    print("="*50)
    
    user_tool = DataTransferTool.create(UserProfile, "LargeDataUserTool")
    agent = Agent(
        system_prompt="Transfer user data with extensive interests list.",
        tools=[user_tool]
    )
    
    print("\n📝 Transferring user with many interests...")
    
    # Create a user with many interests
    many_interests = [
        "programming", "hiking", "photography", "cooking", "reading",
        "travel", "music", "art", "gaming", "fitness", "meditation",
        "gardening", "woodworking", "astronomy", "languages"
    ]
    
    interests_str = str(many_interests).replace("'", '"')
    
    response = await agent.run(
        f"Transfer user: name='Active Person', age=32, "
        f"email='active@hobbies.com', bio='Person with many interests', "
        f"interests={interests_str}"
    )
    
    print(f"\n🤖 Response: {response[:250]}...")
    
    user_data = user_tool.get()
    if user_data:
        print(f"\n✅ Retrieved user with {len(user_data.interests)} interests")
        print(f"   First 5 interests: {user_data.interests[:5]}")
    else:
        print("\n❌ Large data transfer failed")


async def demo_list_of_models():
    """Demonstrate transferring a list of model instances."""
    print("\n" + "="*50)
    print("List of Models Transfer")
    print("="*50)
    
    team_tool = DataTransferTool.create(Team, "TeamTool")
    agent = Agent(
        system_prompt="You transfer team data with a list of team members.",
        tools=[team_tool]
    )
    
    print("\n📝 Transferring team with multiple members...")
    
    response = await agent.run(
        "Transfer team data: team_name='Development Team', "
        "members=[{name:'Alice Johnson', role:'Lead Developer', email:'alice@company.com'}, "
        "{name:'Bob Smith', role:'Frontend Developer', email:'bob@company.com'}, "
        "{name:'Carol Davis', role:'Backend Developer', email:'carol@company.com'}], "
        "project='E-commerce Platform'"
    )
    
    print(f"\n🤖 Response: {response[:300]}...")
    
    team_data = team_tool.get()
    if team_data:
        print(f"\n✅ Retrieved team: {team_data.team_name}")
        print(f"   Project: {team_data.project}")
        print(f"   Members: {len(team_data.members)} people")
        for i, member in enumerate(team_data.members):
            print(f"   {i+1}. {member.name} - {member.role}")
    else:
        print("\n❌ No team data retrieved")


async def demo_dict_of_models():
    """Demonstrate transferring a dictionary with model values."""
    print("\n" + "="*50)
    print("Dictionary of Models Transfer")  
    print("="*50)
    
    company_tool = DataTransferTool.create(Company, "CompanyTool")
    agent = Agent(
        system_prompt="You transfer company data with departments as a dictionary.",
        tools=[company_tool]
    )
    
    print("\n📝 Transferring company with department dictionary...")
    
    response = await agent.run(
        "Transfer company: company_name='Tech Solutions Inc', "
        "departments={"
        "'engineering': {name:'Engineering', budget:500000.0, head:'Alice Johnson', employee_count:15}, "
        "'marketing': {name:'Marketing', budget:200000.0, head:'Bob Wilson', employee_count:8}, "
        "'sales': {name:'Sales', budget:150000.0, head:'Carol Brown', employee_count:12}"
        "}, "
        "founded_year=2018"
    )
    
    print(f"\n🤖 Response: {response[:300]}...")
    
    company_data = company_tool.get()
    if company_data:
        print(f"\n✅ Retrieved company: {company_data.company_name}")
        print(f"   Founded: {company_data.founded_year}")
        print(f"   Departments: {len(company_data.departments)} departments")
        for dept_key, dept in company_data.departments.items():
            print(f"   {dept_key}: {dept.name} ({dept.employee_count} employees, ${dept.budget:,.0f} budget)")
    else:
        print("\n❌ No company data retrieved")


async def run_all():
    """Run all complex scenario demonstrations."""
    print("\n" + "="*60)
    print("COMPLEX SCENARIO DEMOS")
    print("="*60)
    
    await demo_validation_errors()
    await demo_nested_models()
    await demo_field_constraints()
    await demo_large_data()
    await demo_list_of_models()
    await demo_dict_of_models()
    
    print("\n✅ Complex scenario demos completed")