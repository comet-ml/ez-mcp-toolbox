#!/usr/bin/env python3
"""
Utility module for MCP server with automatic parameter generation from Python functions.
"""

import inspect
import json
import importlib.util
import sys
import os
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional, Union
from mcp import Tool
from rich.console import Console
from opik import track
import opik


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
                property_schema["items"] = self._get_array_items_schema(
                    param.annotation
                )

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
                    return {"type": "array", "items": {"type": inner_type}}
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
            filtered_arguments = {
                k: v for k, v in arguments.items() if k in func_params
            }

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
    skip_functions = {"TypedDict"}

    # First, look for standalone functions (backward compatibility)
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isfunction(obj)
            and not name.startswith("_")
            and name not in skip_functions
        ):
            # Register the function as a tool
            registry.tool(obj)

    # Then, look for classes with methods that can be used as tools
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and not name.startswith("_")
            and name not in skip_functions
            and name in module.__dict__
            and obj.__module__ == module.__name__
        ):

            # Create an instance of the class
            try:
                instance = obj()

                # Find all methods in the class - only those defined in the class itself
                for method_name, method in instance.__class__.__dict__.items():
                    if (
                        not method_name.startswith("_")
                        and method_name not in skip_functions
                        and callable(method)
                        and inspect.isfunction(method)
                    ):

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


# =============================================================================
# Common LLM Processing and Opik Utilities
# =============================================================================


def configure_opik(opik_mode: str = "hosted", project_name: str = "ez-mcp-toolbox"):
    """
    Configure Opik based on the specified mode.

    Args:
        opik_mode: Opik mode - "local", "hosted", or "disabled"
        project_name: Project name for Opik tracking
    """
    if opik_mode == "disabled":
        return

    # Set the project name via environment variable
    os.environ["OPIK_PROJECT_NAME"] = project_name

    # Check if ~/.opik.config file exists
    opik_config_path = os.path.expanduser("~/.opik.config")
    if os.path.exists(opik_config_path):
        print("✅ Found existing ~/.opik.config file, skipping opik.configure()")
        return

    try:
        if opik_mode == "local":
            opik.configure(use_local=True)
        elif opik_mode == "hosted":
            # For hosted mode, Opik will use environment variables or default configuration
            opik.configure(use_local=False)
        else:
            print(f"Warning: Unknown Opik mode '{opik_mode}', using hosted mode")
            opik.configure(use_local=False)

        # Note: We don't use LiteLLM's OpikLogger as it creates separate traces
        # Instead, we'll manually manage spans within the existing trace
        print("✅ Opik configured for manual span management")

    except Exception as e:
        print(f"Warning: Opik configuration failed: {e}")
        print("Continuing without Opik tracing...")


@track(name="llm_completion", type="llm")
def call_llm_with_tracing(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False,
    console: Optional[Console] = None,
    **kwargs,
):
    """
    Call LLM with proper Opik span management.

    Args:
        model: LLM model to use
        messages: List of messages for the LLM
        tools: Optional list of tools for the LLM
        debug: Whether to enable debug output
        console: Rich console for output (optional)
        **kwargs: Additional arguments for the LLM call

    Returns:
        LLM response object
    """
    from litellm import completion

    try:
        if debug:
            if console:
                console.print(f"🤖 Calling LLM: {model}")
                console.print(f"📝 Messages count: {len(messages)}")
                console.print(f"🔧 Tools count: {len(tools) if tools else 0}")
                if kwargs:
                    console.print(f"⚙️  Model kwargs: {kwargs}")
            else:
                print(f"🤖 Calling LLM: {model}")
                print(f"📝 Messages count: {len(messages)}")
                print(f"🔧 Tools count: {len(tools) if tools else 0}")
                if kwargs:
                    print(f"⚙️  Model kwargs: {kwargs}")

        # Call the LLM - Opik will automatically track this as a span within the current trace
        call_kwargs = kwargs.copy()
        if tools:
            call_kwargs.update({"tools": tools, "tool_choice": "auto"})

        resp = completion(
            model=model,
            messages=messages,
            **call_kwargs,
        )

        if debug:
            if console:
                console.print(f"📊 LLM response type: {type(resp)}")
            else:
                print(f"📊 LLM response type: {type(resp)}")

        if resp is None:
            if debug:
                if console:
                    console.print("❌ LLM returned None response")
                else:
                    print("❌ LLM returned None response")
            raise ValueError("LLM returned None response")

        if not hasattr(resp, "choices"):
            if debug:
                if console:
                    console.print(
                        f"❌ LLM response missing 'choices' attribute: {resp}"
                    )
                else:
                    print(f"❌ LLM response missing 'choices' attribute: {resp}")
            raise ValueError(f"LLM response missing 'choices' attribute: {resp}")

        if debug:
            if console:
                console.print(f"✅ LLM response has {len(resp.choices)} choices")
            else:
                print(f"✅ LLM response has {len(resp.choices)} choices")

        return resp

    except Exception as e:
        if debug:
            if console:
                console.print(f"❌ LLM call failed: {e}")
                console.print(f"❌ Exception type: {type(e)}")
            else:
                print(f"❌ LLM call failed: {e}")
                print(f"❌ Exception type: {type(e)}")
        raise


def extract_llm_content(resp, debug: bool = False, console: Optional[Console] = None):
    """
    Extract content from LLM response, handling both text and tool call responses.

    Args:
        resp: LLM response object
        debug: Whether to enable debug output
        console: Rich console for output (optional)

    Returns:
        Tuple of (content, tool_calls)
    """
    if not resp or not hasattr(resp, "choices"):
        return None, None

    choice = resp.choices[0].message
    content = getattr(choice, "content", None)
    tool_calls = getattr(choice, "tool_calls", None)

    if debug:
        if console:
            if tool_calls:
                console.print(f"🔧 LLM requested {len(tool_calls)} tool calls")
            else:
                console.print(
                    f"✅ LLM returned text response: {len(content or '')} characters"
                )
        else:
            if tool_calls:
                print(f"🔧 LLM requested {len(tool_calls)} tool calls")
            else:
                print(f"✅ LLM returned text response: {len(content or '')} characters")

    return content, tool_calls


def create_llm_messages(
    system_prompt: str,
    user_input: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Create properly formatted messages for LLM calls.

    Args:
        system_prompt: System prompt for the LLM
        user_input: User input text
        conversation_history: Optional conversation history

    Returns:
        List of formatted messages
    """
    messages = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_input})

    return messages


def format_tool_result(tool_call_id: str, content: str) -> Dict[str, Any]:
    """
    Format tool execution result for LLM consumption.

    Args:
        tool_call_id: ID of the tool call
        content: Result content from tool execution

    Returns:
        Formatted tool result message
    """
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def format_assistant_tool_calls(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format assistant tool calls for conversation history.

    Args:
        tool_calls: List of tool calls made by assistant

    Returns:
        Formatted assistant message with tool calls
    """
    return {"role": "assistant", "tool_calls": tool_calls, "content": ""}
