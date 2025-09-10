#!/usr/bin/env python3
# datatransfer.py - Generic data transfer tool using Pydantic BaseModel

import json
from typing import Generic, TypeVar, Type, Optional, Dict, Any, Union
from pydantic import BaseModel, ValidationError

from claude_agent_toolkit import BaseTool, tool

# Generic type variable bound to Pydantic BaseModel
T = TypeVar('T', bound=BaseModel)


class DataTransferTool(BaseTool, Generic[T]):
    """
    A generic data transfer tool that can work with any Pydantic BaseModel.
    
    This tool enables type-safe data transfer between Claude agents and host applications
    by using Pydantic models for validation and serialization. The tool description
    automatically includes the model schema so Claude understands the expected data structure.
    
    Features:
    - Generic implementation works with any Pydantic BaseModel subclass
    - Automatic schema inclusion in tool description for Claude
    - Type-safe data validation and transfer
    - Simple transfer/get interface for host applications
    - Runtime validation with clear error messages
    
    Usage:
        # Define your data model
        class UserData(BaseModel):
            name: str
            age: int
            email: str
            
        # Create tool instance for specific model
        user_tool = DataTransferTool.create(UserData)
        
        # Use with agent
        agent = Agent(tools=[user_tool])
        await agent.run("Transfer user data: name='John', age=30, email='john@example.com'")
        
        # Retrieve transferred data from host
        user_data = user_tool.get()
        if user_data:
            print(f"Received: {user_data.name}, {user_data.age}, {user_data.email}")
    """
    
    def __init__(self, model_class: Type[T], *, workers: Optional[int] = None, log_level: str = "ERROR"):
        """
        Initialize the DataTransferTool for a specific Pydantic model.
        
        Args:
            model_class: The Pydantic BaseModel class to use for data validation
            workers: Number of worker processes (for parallel operations)
            log_level: Logging level for FastMCP (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # Store model class and transferred data
        self._model_class: Type[T] = model_class
        self._transferred_data: Optional[T] = None
        
        # Generate schema information for the tool description
        self._schema = model_class.model_json_schema()
        
        # Update the tool description BEFORE calling BaseTool.__init__
        # This ensures FastMCP registers the tool with the correct description
        self._update_tool_description()
        
        # NOW initialize the BaseTool (which starts the MCP server)
        # Auto host/port selection, only accept workers and log_level
        super().__init__(workers=workers, log_level=log_level)
    
    @classmethod
    def create(cls, model_class: Type[T], name: Optional[str] = None) -> "DataTransferTool[T]":
        """
        Factory method to create a DataTransferTool instance for a specific model.
        
        Args:
            model_class: The Pydantic BaseModel class to use
            name: Optional custom name for the tool class (defaults to model_class.__name__ + "TransferTool")
            
        Returns:
            DataTransferTool instance configured for the specified model with distinct class name
            
        Example:
            user_tool = DataTransferTool.create(UserData)  # Creates UserDataTransferTool
            product_tool = DataTransferTool.create(ProductInfo, "ProductTool")  # Creates ProductTool
        """
        # Generate class name
        if name is None:
            class_name = f"{model_class.__name__}TransferTool"
        else:
            class_name = name
        
        # Create a new class dynamically with the specified name
        # This makes each tool appear as a distinct class to Claude
        DynamicClass = type(class_name, (cls,), {
            '__module__': cls.__module__,
            '__qualname__': class_name,
        })
        
        # Create instance using normal constructor
        return DynamicClass(model_class)
    
    def _update_tool_description(self) -> None:
        """Update the transfer tool description with model schema information."""
        model_name = self._model_class.__name__
        
        # Extract field information from schema
        properties = self._schema.get('properties', {})
        required_fields = self._schema.get('required', [])
        
        # Build field descriptions
        field_descriptions = []
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'unknown')
            field_desc = field_info.get('description', '')
            required_marker = " (required)" if field_name in required_fields else " (optional)"
            
            field_line = f"  - {field_name}: {field_type}{required_marker}"
            if field_desc:
                field_line += f" - {field_desc}"
            field_descriptions.append(field_line)
        
        # Get the actual class name for the tool
        tool_class_name = self.__class__.__name__
        
        # Create comprehensive tool description with clear identity
        description = f"""🎯 {tool_class_name}: Transfer {model_name} data exclusively.

⚠️  IMPORTANT: This tool ({tool_class_name}) ONLY accepts {model_name} data structures. Do not use for other data types.

📋 Required Schema for {model_name}:
{chr(10).join(field_descriptions)}

✅ Usage: Call transfer() with data matching the {model_name} structure:
Example: transfer(data={{"field1": "value1", "field2": "value2"}})

📖 Complete JSON Schema for {model_name}:
{json.dumps(self._schema, indent=2)}

🔧 Tool Identity: {tool_class_name} - configured for {model_name} only"""
        
        # Dynamically update the tool method description
        if hasattr(self.transfer, '__mcp_meta__'):
            self.transfer.__mcp_meta__['description'] = description
    
    @tool(description="Transfer structured data - description will be updated dynamically")
    async def transfer(self, data: Union[Dict[str, Any], T]) -> Dict[str, Any]:
        """
        Transfer and validate data according to the configured model schema.
        
        Args:
            data: Data to transfer, either as a dict or model instance
            
        Returns:
            Dict containing transfer status and data summary
        """
        try:
            # If data is already a model instance, use it directly
            if isinstance(data, self._model_class):
                validated_data = data
            elif isinstance(data, dict):
                # Validate and parse the data using the model class
                validated_data = self._model_class(**data)
            else:
                # Try to convert other types to dict first
                try:
                    if hasattr(data, 'model_dump'):
                        # Handle other Pydantic models
                        dict_data = data.model_dump()
                    else:
                        # Try direct conversion
                        dict_data = dict(data)
                    validated_data = self._model_class(**dict_data)
                except (TypeError, ValueError) as e:
                    return {
                        'success': False,
                        'error': f'Cannot convert data to {self._model_class.__name__}: {str(e)}',
                        'received_type': type(data).__name__
                    }
            
            # Store the validated data
            self._transferred_data = validated_data
            
            # Create summary of transferred data
            data_summary = validated_data.model_dump()
            
            return {
                'success': True,
                'message': f'Successfully transferred {self._model_class.__name__} data',
                'model_type': self._model_class.__name__,
                'data_summary': data_summary,
                'field_count': len(data_summary)
            }
            
        except ValidationError as e:
            return {
                'success': False,
                'error': f'Validation failed for {self._model_class.__name__}',
                'validation_errors': e.errors(),
                'received_data': data if isinstance(data, dict) else str(data)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error during data transfer: {str(e)}',
                'model_type': self._model_class.__name__
            }
    
    def get(self) -> Optional[T]:
        """
        Retrieve the transferred data from the host application.
        
        Returns:
            The transferred and validated model instance, or None if no data has been transferred
            
        Example:
            user_data = user_tool.get()
            if user_data:
                print(f"Name: {user_data.name}")
        """
        return self._transferred_data
    
    def has_data(self) -> bool:
        """
        Check if data has been transferred.
        
        Returns:
            True if data has been transferred, False otherwise
        """
        return self._transferred_data is not None
    
    def clear(self) -> None:
        """
        Clear the transferred data.
        
        This can be useful for resetting the tool state or preparing for new data transfer.
        """
        self._transferred_data = None
    
    def get_model_class(self) -> Type[T]:
        """
        Get the Pydantic model class this tool is configured for.
        
        Returns:
            The Pydantic BaseModel class
        """
        return self._model_class
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the configured model.
        
        Returns:
            The JSON schema dict for the model
        """
        return self._schema.copy()
    
    def to_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get the transferred data as a dictionary.
        
        Returns:
            Dict representation of transferred data, or None if no data
        """
        if self._transferred_data is None:
            return None
        return self._transferred_data.model_dump()
    
    def to_json(self) -> Optional[str]:
        """
        Get the transferred data as a JSON string.
        
        Returns:
            JSON string representation of transferred data, or None if no data
        """
        if self._transferred_data is None:
            return None
        return self._transferred_data.model_dump_json()