#!/usr/bin/env python3
"""
Utility module for MCP server with automatic parameter generation from Python functions.
"""

import inspect
import json
import importlib.util
import sys
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional, Union
from mcp import Tool


class ToolRegistry:
    """Registry for managing MCP tools with automatic parameter generation."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, func_or_name=None, description: Optional[str] = None):
        """
        Decorator to register a function as an MCP tool.

        Can be used in two ways:
        1. @tool - uses function name and docstring
        2. @tool("custom_name") or @tool(description="custom description")
        """

        def decorator(func: Callable) -> Callable:
            # Determine if first argument is a function (no parentheses) or name/description
            if callable(func_or_name):
                # Used as @tool (no parentheses)
                tool_name = func_or_name.__name__
                tool_description = func_or_name.__doc__ or f"Tool: {tool_name}"
                func = func_or_name
            else:
                # Used as @tool("name") or @tool(description="desc")
                tool_name = (
                    func_or_name if isinstance(func_or_name, str) else func.__name__
                )
                tool_description = description or func.__doc__ or f"Tool: {tool_name}"

            # Generate input schema from function signature
            input_schema = self._generate_input_schema(func)

            self._tools[tool_name] = {
                "function": func,
                "description": tool_description,
                "input_schema": input_schema,
            }

            return func

        # If called without parentheses (@tool), func_or_name is the function
        if callable(func_or_name):
            return decorator(func_or_name)
        else:
            # If called with parentheses (@tool(...)), return the decorator
            return decorator

    def _generate_input_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate JSON schema from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":  # Skip self parameter
                continue

            # Determine parameter type
            param_type = self._get_json_type(param.annotation)

            # Get parameter description from docstring or default
            description = self._get_param_description(func, param_name)

            # Create the property schema
            property_schema = {"type": param_type, "description": description}
            
            # Handle array types - add items schema
            if param_type == "array":
                property_schema["items"] = self._get_array_items_schema(param.annotation)

            properties[param_name] = property_schema

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {"type": "object", "properties": properties, "required": required}

    def _get_json_type(self, annotation: Any) -> str:
        """Convert Python type annotation to JSON schema type."""
        if annotation == inspect.Parameter.empty:
            return "string"  # Default type

        # Handle typing types
        if hasattr(annotation, "__origin__"):
            if annotation.__origin__ is Union:
                # For Union types, use the first non-None type
                args = annotation.__args__
                non_none_args = [arg for arg in args if arg != type(None)]
                if non_none_args:
                    return self._get_json_type(non_none_args[0])
                return "string"
            elif annotation.__origin__ is list:
                # Handle List[str], List[int], etc.
                return "array"

        # Handle basic types
        type_mapping = {
            int: "number",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_mapping.get(annotation, "string")

    def _get_array_items_schema(self, annotation: Any) -> Dict[str, Any]:
        """Generate items schema for array types."""
        if hasattr(annotation, "__args__") and annotation.__args__:
            # Handle List[SomeType] - get the type of items
            item_type = annotation.__args__[0]
            
            # Handle nested List types like List[List[float]]
            if hasattr(item_type, "__origin__") and item_type.__origin__ is list:
                # For List[List[SomeType]], return array of arrays
                if item_type.__args__:
                    inner_type = self._get_json_type(item_type.__args__[0])
                    return {
                        "type": "array",
                        "items": {"type": inner_type}
                    }
                else:
                    return {"type": "array", "items": {"type": "string"}}
            else:
                # For List[SomeType], return the type of items
                inner_type = self._get_json_type(item_type)
                return {"type": inner_type}
        else:
            # Fallback for generic List
            return {"type": "string"}

    def _get_param_description(self, func: Callable, param_name: str) -> str:
        """Extract parameter description from function docstring."""
        doc = func.__doc__
        if not doc:
            return f"Parameter: {param_name}"

        # Simple parsing of docstring for parameter descriptions
        lines = doc.strip().split("\n")
        for line in lines:
            line = line.strip()
            if (
                line.startswith(f"{param_name}:")
                or line.startswith(f"Args:")
                and param_name in line
            ):
                # Extract description after colon
                if ":" in line:
                    return line.split(":", 1)[1].strip()

        return f"Parameter: {param_name}"

    def get_tools(self) -> List[Tool]:
        """Get list of MCP Tool objects."""
        tools = []
        for name, tool_info in self._tools.items():
            tools.append(
                Tool(
                    name=name,
                    description=tool_info["description"],
                    inputSchema=tool_info["input_schema"],
                )
            )
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call a tool by name with given arguments."""
        if name not in self._tools:
            return [{"type": "text", "text": f"Unknown tool: {name}"}]

        try:
            func = self._tools[name]["function"]
            
            # Get the function signature to determine which arguments to pass
            sig = inspect.signature(func)
            func_params = set(sig.parameters.keys())
            
            # Filter arguments to only include those that the function accepts
            filtered_arguments = {k: v for k, v in arguments.items() if k in func_params}
            
            result = func(**filtered_arguments)

            # Convert result to MCP format
            if isinstance(result, str):
                return [{"type": "text", "text": result}]
            elif isinstance(result, (dict, list)):
                # For structured data, return as JSON
                return [{"type": "text", "text": json.dumps(result, indent=2)}]
            else:
                return [{"type": "text", "text": str(result)}]

        except Exception as e:
            return [{"type": "text", "text": f"Error calling tool {name}: {str(e)}"}]


# Global registry instance
registry = ToolRegistry()


# Tool decorator for easy registration
def tool(func_or_name=None, description: Optional[str] = None):
    """Decorator to register a function as an MCP tool."""
    return registry.tool(func_or_name, description)


def load_tools_from_file(file_path: str) -> None:
    """
    Load tools from a Python file and register them with the global registry.
    Supports both standalone functions and class methods.
    
    Args:
        file_path: Path to the Python file containing tool functions or classes
    """
    # Clear existing tools
    registry._tools.clear()
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location("tools_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["tools_module"] = module
    spec.loader.exec_module(module)
    
    # Skip utility functions and classes
    skip_functions = {
        'TypedDict'
    }
    
    # First, look for standalone functions (backward compatibility)
    for name, obj in inspect.getmembers(module):
        if (inspect.isfunction(obj) and 
            not name.startswith('_') and 
            name not in skip_functions):
            # Register the function as a tool
            registry.tool(obj)
    
    # Then, look for classes with methods that can be used as tools
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            not name.startswith('_') and 
            name not in skip_functions and
            name in module.__dict__ and
            obj.__module__ == module.__name__):
            
            # Create an instance of the class
            try:
                instance = obj()
                
                # Find all methods in the class - only those defined in the class itself
                for method_name, method in instance.__class__.__dict__.items():
                    if (not method_name.startswith('_') and 
                        method_name not in skip_functions and
                        callable(method) and
                        inspect.isfunction(method)):
                        
                        # Create a wrapper function that calls the method
                        def create_method_wrapper(inst, meth):
                            # Get the original method signature
                            original_sig = inspect.signature(meth)
                            
                            def wrapper(*args, **kwargs):
                                return meth(inst, *args, **kwargs)
                            
                            # Preserve the original method signature
                            wrapper.__signature__ = original_sig
                            return wrapper
                        
                        wrapper_func = create_method_wrapper(instance, method)
                        wrapper_func.__name__ = method_name
                        wrapper_func.__doc__ = method.__doc__
                        
                        # Register the wrapper as a tool
                        registry.tool(wrapper_func)
                        
            except Exception as e:
                print(f"Warning: Could not instantiate class {name}: {e}")
                continue
    
    print(f"Loaded {len(registry._tools)} tools from {file_path}")
