#!/usr/bin/env python3
# models.py - Shared Pydantic models for DataTransfer demos

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


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


class CustomerInfo(BaseModel):
    """Customer information model."""
    customer_id: str = Field(..., description="Unique customer identifier")
    name: str = Field(..., description="Customer full name")
    email: str = Field(..., description="Customer email address")
    phone: Optional[str] = Field(None, description="Customer phone number")


class OrderItem(BaseModel):
    """Order item model."""
    product_name: str = Field(..., description="Name of the product")
    quantity: int = Field(..., gt=0, description="Quantity ordered")
    unit_price: float = Field(..., gt=0, description="Price per unit")


class Order(BaseModel):
    """Order model with nested items."""
    order_id: str = Field(..., description="Unique order identifier")
    customer: CustomerInfo = Field(..., description="Customer information")
    items: List[OrderItem] = Field(..., description="List of ordered items")
    total_amount: float = Field(..., gt=0, description="Total order amount")
    status: str = Field(default="pending", description="Order status")


class Configuration(BaseModel):
    """Configuration model with various field types."""
    name: str = Field(..., description="Configuration name")
    enabled: bool = Field(True, description="Whether configuration is enabled")
    timeout_seconds: int = Field(30, ge=1, le=3600, description="Timeout in seconds")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retries")
    allowed_hosts: List[str] = Field(default_factory=list, description="List of allowed hostnames")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class TeamMember(BaseModel):
    """Team member model for list demo."""
    name: str = Field(..., description="Team member name")
    role: str = Field(..., description="Team member role")
    email: str = Field(..., description="Team member email")


class Team(BaseModel):
    """Team model containing list of members."""
    team_name: str = Field(..., description="Team name")
    members: List[TeamMember] = Field(..., description="List of team members")
    project: str = Field(..., description="Current project")


class Department(BaseModel):
    """Department model for dictionary demo."""
    name: str = Field(..., description="Department name")
    budget: float = Field(..., gt=0, description="Department budget")
    head: str = Field(..., description="Department head")
    employee_count: int = Field(..., gt=0, description="Number of employees")


class Company(BaseModel):
    """Company model containing dictionary of departments."""
    company_name: str = Field(..., description="Company name")
    departments: Dict[str, Department] = Field(..., description="Departments by key")
    founded_year: int = Field(..., gt=1800, description="Year founded")