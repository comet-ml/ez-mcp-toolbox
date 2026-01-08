#!/usr/bin/env python3
"""
Utility module for MCP server with automatic parameter generation from Python functions.
"""

import inspect
import json
import importlib.util
import sys
import os
import warnings
import tempfile
import urllib.request
import urllib.parse
import atexit
from typing import Any, Dict, List, Callable, Optional, Union

from mcp import Tool
from rich.console import Console
from opik import track, Opik, opik_context
from opik.evaluation import metrics as opik_metrics
import opik
import litellm
from litellm.utils import get_llm_provider

# Suppress litellm RuntimeWarning about coroutines never awaited
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="coroutine 'close_litellm_async_clients' was never awaited",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*"
)
# Suppress pydantic serialization warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Pydantic.*serializer.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*PydanticSerialization.*",
)

# Configure litellm to drop params
litellm.drop_params = True


def extract_provider_from_model(model: str) -> tuple[Optional[str], str]:
    """
    Extract provider name from model string and return normalized model name.

    Handles formats like:
    - "openai/gpt-5-nano" -> ("openai", "gpt-5-nano")
    - "gpt-3.5-turbo" -> (None or provider from litellm, "gpt-3.5-turbo")
    - "anthropic/claude-3-opus" -> ("anthropic", "claude-3-opus")

    Args:
        model: Model string that may contain provider prefix

    Returns:
        Tuple of (provider_name, normalized_model_name)
    """
    if not model:
        return None, model

    # Check if model string has provider prefix (format: "provider/model-name")
    if "/" in model:
        provider_part, model_part = model.split("/", 1)
        provider_part = provider_part.strip()
        model_part = model_part.strip()
        # Common provider names (normalize to lowercase)
        provider_lower = provider_part.lower()
        known_providers = {
            "openai",
            "anthropic",
            "google",
            "cohere",
            "mistral",
            "azure",
            "bedrock",
            "vertex",
            "together",
            "replicate",
            "huggingface",
            "ollama",
            "groq",
            "deepinfra",
            "anyscale",
            "perplexity",
            "voyage",
            "nvidia",
            "cloudflare",
            "cerebras",
            "predibase",
        }
        if provider_lower in known_providers:
            return provider_lower, model_part

    # Try using litellm's get_llm_provider function
    provider = None
    try:
        _, provider, _ = get_llm_provider(model=model)
        if provider:
            provider = provider.lower() if isinstance(provider, str) else None
    except Exception:
        pass

    # Fallback: Map common model names to providers
    if not provider:
        model_lower = model.lower()
        if model_lower.startswith("gpt") or "gpt" in model_lower:
            provider = "openai"
        elif model_lower.startswith("claude") or "claude" in model_lower:
            provider = "anthropic"
        elif model_lower.startswith("gemini") or "gemini" in model_lower:
            provider = "google"
        elif model_lower.startswith("llama") or "llama" in model_lower:
            # Llama models could be from various providers, but default to meta if unsure
            pass  # Keep as None since provider is unclear

    return provider, model


def update_opik_span_and_trace_with_usage(model: str, resp: Any) -> None:
    """
    Extract token counts and usage information from LLM response and update Opik span and trace.

    This function:
    1. Extracts provider and normalizes model name
    2. Extracts token counts (prompt_tokens, completion_tokens, total_tokens) from response
    3. Updates the current span with provider, model, and usage
    4. Updates trace metadata with provider and model (usage is aggregated by Opik)

    Args:
        model: Model string (may contain provider prefix like "openai/gpt-5-nano")
        resp: LLM response object from litellm.completion()
    """
    try:
        # Extract provider from model string (with fallback parsing) and normalize model name
        provider, normalized_model = extract_provider_from_model(model)

        # Extract token counts from response
        usage_dict = {}
        if hasattr(resp, "usage") and resp.usage:
            prompt_tokens = getattr(resp.usage, "prompt_tokens", None)
            completion_tokens = getattr(resp.usage, "completion_tokens", None)
            total_tokens = getattr(resp.usage, "total_tokens", None)

            if prompt_tokens is not None:
                usage_dict["prompt_tokens"] = prompt_tokens
            if completion_tokens is not None:
                usage_dict["completion_tokens"] = completion_tokens
            if total_tokens is not None:
                usage_dict["total_tokens"] = total_tokens

        # Update span with provider, model, and usage information
        # Always include provider if available, and always include usage if present
        if usage_dict:
            update_kwargs = {
                "model": normalized_model,  # Use normalized model (without provider prefix)
                "usage": usage_dict,
            }
            # Always set provider if we could determine it (required for cost estimation)
            if provider:
                update_kwargs["provider"] = provider

            try:
                opik_context.update_current_span(**update_kwargs)

                # Also update trace metadata with provider and model for cost estimation
                # Note: Usage is aggregated automatically by Opik from spans, we just need provider/model
                try:
                    # Get existing metadata from trace data - always fetch fresh to avoid overwriting
                    existing_metadata = {}
                    try:
                        trace_data = opik_context.get_current_trace_data()

                        # Handle TraceData object (not just dict)
                        if trace_data:
                            # Check if it's a dict
                            if isinstance(trace_data, dict):
                                metadata = trace_data.get("metadata")
                                if isinstance(metadata, dict):
                                    existing_metadata = metadata.copy()
                                else:
                                    existing_metadata = {}
                            # Check if it's a TraceData object with metadata attribute
                            elif hasattr(trace_data, "metadata"):
                                metadata = getattr(trace_data, "metadata", None)
                                if isinstance(metadata, dict):
                                    existing_metadata = metadata.copy()
                                else:
                                    existing_metadata = {}
                    except Exception:
                        existing_metadata = {}

                    # Add provider and model to metadata if available (preserve existing metadata)
                    if provider:
                        existing_metadata["provider"] = provider
                    if normalized_model:
                        existing_metadata["model"] = normalized_model

                    # Update trace metadata with provider and model
                    # Usage will be aggregated automatically by Opik from spans
                    if provider or normalized_model:
                        opik_context.update_current_trace(metadata=existing_metadata)
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass


class ToolRegistry:
    """Registry for managing MCP tools with automatic parameter generation."""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, func_or_name: Any = None, description: Optional[str] = None) -> Any:
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
            property_schema: Dict[str, Any] = {
                "type": param_type,
                "description": description,
            }

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
                or line.startswith("Args:")
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
                # For structured data, preserve the structure by wrapping in a special format
                return [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"__structured_data__": result}, separators=(",", ":")
                        ),
                    }
                ]
            else:
                return [{"type": "text", "text": str(result)}]

        except Exception as e:
            return [{"type": "text", "text": f"Error calling tool {name}: {str(e)}"}]

    def filter_tools(
        self,
        include_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
    ) -> None:
        """
        Filter tools based on include and exclude regex patterns.

        Args:
            include_pattern: Python regex pattern to include only matching tool names
            exclude_pattern: Python regex pattern to exclude matching tool names
        """
        import re

        tools_to_remove = []

        for tool_name in self._tools.keys():
            should_remove = False

            # Apply include filter
            if include_pattern:
                if not re.search(include_pattern, tool_name):
                    should_remove = True

            # Apply exclude filter
            if exclude_pattern:
                if re.search(exclude_pattern, tool_name):
                    should_remove = True

            if should_remove:
                tools_to_remove.append(tool_name)

        # Remove filtered tools
        for tool_name in tools_to_remove:
            del self._tools[tool_name]


# Global registry instance
registry = ToolRegistry()


# Tool decorator for easy registration
def tool(func_or_name: Any = None, description: Optional[str] = None) -> Any:
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

    # Check for quiet mode (from environment variable)
    quiet_mode = os.getenv("EZ_MCP_QUIET") == "1"

    # Check for and call _initialize function if it exists
    if hasattr(module, "_initialize") and callable(module._initialize):
        try:
            module._initialize()
        except Exception as e:
            if not quiet_mode:
                print(f"Warning: Error calling _initialize function: {e}")

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
                        def create_method_wrapper(inst: Any, meth: Any) -> Any:
                            # Get the original method signature
                            original_sig = inspect.signature(meth)

                            def wrapper(*args: Any, **kwargs: Any) -> Any:
                                return meth(inst, *args, **kwargs)

                            # Preserve the original method signature
                            wrapper.__signature__ = original_sig  # type: ignore
                            return wrapper

                        wrapper_func = create_method_wrapper(instance, method)
                        wrapper_func.__name__ = method_name
                        wrapper_func.__doc__ = method.__doc__

                        # Register the wrapper as a tool
                        registry.tool(wrapper_func)

            except Exception as e:
                if not quiet_mode:
                    print(f"Warning: Could not instantiate class {name}: {e}")
                continue

    if not quiet_mode:
        print(f"Loaded {len(registry._tools)} tools from {file_path}")


def load_tools_from_module(module_name: str) -> None:
    """
    Load tools from a Python module by name and register them with the global registry.
    Supports both standalone functions and class methods.

    Args:
        module_name: Name of the module to load (e.g., 'opik_optimizer.utils.core')
    """
    # Clear existing tools
    registry._tools.clear()

    try:
        # Import the module
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")

    # Check for quiet mode (from environment variable)
    quiet_mode = os.getenv("EZ_MCP_QUIET") == "1"

    # Check for and call _initialize function if it exists
    if hasattr(module, "_initialize") and callable(module._initialize):
        try:
            module._initialize()
        except Exception as e:
            if not quiet_mode:
                print(f"Warning: Error calling _initialize function: {e}")

    # Skip utility functions and classes
    skip_functions = {"TypedDict"}

    # First, look for standalone functions
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
                        def create_method_wrapper(inst: Any, meth: Any) -> Any:
                            # Get the original method signature
                            original_sig = inspect.signature(meth)

                            def wrapper(*args: Any, **kwargs: Any) -> Any:
                                return meth(inst, *args, **kwargs)

                            # Preserve the original method signature
                            wrapper.__signature__ = original_sig  # type: ignore
                            return wrapper

                        wrapper_func = create_method_wrapper(instance, method)
                        wrapper_func.__name__ = method_name
                        wrapper_func.__doc__ = method.__doc__

                        # Register the wrapper as a tool
                        registry.tool(wrapper_func)

            except Exception as e:
                if not quiet_mode:
                    print(f"Warning: Could not instantiate class {name}: {e}")
                continue

    if not quiet_mode:
        print(f"Loaded {len(registry._tools)} tools from module {module_name}")


# =============================================================================
# Common LLM Processing and Opik Utilities
# =============================================================================


def configure_opik(
    opik_mode: str = "hosted", project_name: str = "ez-mcp-toolbox"
) -> None:
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


async def generate_mcp_mermaid_diagram(mcp_manager: Any) -> Optional[str]:
    """
    Generate a mermaid diagram showing MCP servers and their tools.

    Args:
        mcp_manager: The MCP manager instance

    Returns:
        Mermaid diagram as a string, or None if unable to generate
    """
    try:
        if not mcp_manager or not mcp_manager.sessions:
            return None

        diagram_lines = ["block-beta"]
        diagram_lines.append("    columns 1")  # Stack server and tools vertically
        server_node_ids = {}
        server_classes = []

        # Process each server
        for idx, (server_name, session) in enumerate(mcp_manager.sessions.items()):
            # Create a sanitized node ID for the server (mermaid doesn't like special chars)
            server_node_id = f"server_{idx}"
            tools_block_id = f"{server_node_id}_tools"
            server_node_ids[server_name] = server_node_id

            # Add server node with round borders using parentheses
            sanitized_server_name = server_name.replace('"', '\\"')
            diagram_lines.append(f'    {server_node_id}("📡 {sanitized_server_name}")')
            server_classes.append(server_node_id)

            # Add space before tools block
            diagram_lines.append("    space")

            # Get tools for this server
            try:
                tools_resp = await session.list_tools()
                if tools_resp.tools:
                    # Create a block for tools with column layout (space for label)
                    diagram_lines.append(f'    block:{tools_block_id}[" "]')

                    # Add classDef inside the tools block
                    diagram_lines.append(
                        "        classDef serverStyle fill:#E8F4F8,stroke:#4A90E2,stroke-width:2px"
                    )
                    diagram_lines.append("        columns 4")  # 4 columns

                    # Add each tool as a separate node with round borders
                    for tool_idx, tool in enumerate(tools_resp.tools):
                        tool_node_id = f"{tools_block_id}_tool_{tool_idx}"
                        sanitized_tool_name = f"🔧 {tool.name}".replace('"', '\\"')
                        diagram_lines.append(
                            f'        {tool_node_id}("{sanitized_tool_name}")'
                        )

                    diagram_lines.append("    end")

                    # Connect server to tools block
                    diagram_lines.append(f"    {server_node_id} --> {tools_block_id}")
            except Exception:
                # If we can't get tools, skip the tools block
                pass

        if len(diagram_lines) == 1:
            # Only the graph TB line, no servers
            return None

        # Apply server styling
        if server_classes:
            for server_id in server_classes:
                diagram_lines.append(f"    class {server_id} serverStyle")

        return "\n".join(diagram_lines)

    except Exception:
        # If anything fails, return None to gracefully skip diagram generation
        return None


@track(name="llm_completion", type="llm")
def call_llm_with_tracing(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False,
    console: Optional[Console] = None,
    **kwargs: Any,
) -> Any:
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
            if debug:
                print(f"🔧 Added tools to call_kwargs: {len(tools)} tools")
                print("🔧 Tool choice: auto")
        else:
            if debug:
                print("⚠️  No tools provided to LLM call")

        if debug:
            print(f"🔧 Final call_kwargs keys: {list(call_kwargs.keys())}")

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

        # Extract token counts and update span with usage information
        update_opik_span_and_trace_with_usage(model, resp)

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


def extract_llm_content(
    resp: Any, debug: bool = False, console: Optional[Console] = None
) -> tuple[Optional[str], Optional[Any]]:
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


def format_tool_result(tool_call_id: str, content: Any) -> Dict[str, Any]:
    """
    Format tool execution result for LLM consumption.

    Args:
        tool_call_id: ID of the tool call
        content: Result content from tool execution (can be str, dict, list, or other)

    Returns:
        Formatted tool result message
    """
    # Handle structured data by converting to JSON string for LLM consumption
    if isinstance(content, (dict, list)):
        # For structured data, convert to JSON string so LLM can parse it
        # The LLM will understand this is structured data and can work with it
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(content, indent=2),
        }
    else:
        # For other types, convert to string
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(content)}


def format_assistant_tool_calls(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format assistant tool calls for conversation history.

    Args:
        tool_calls: List of tool calls made by assistant

    Returns:
        Formatted assistant message with tool calls
    """
    return {"role": "assistant", "tool_calls": tool_calls, "content": ""}


def process_mcp_tool_result(
    result: Any, tool_name: Optional[str] = None, debug: bool = False
) -> Any:
    """
    Process MCP tool result, preserving structured data.

    This function extracts and processes the content from an MCP tool result,
    preserving structured data (dict/list) while handling text content appropriately.

    Args:
        result: MCP tool result object
        tool_name: Name of the tool (for debug logging)
        debug: Whether to enable debug output

    Returns:
        Processed result - can be str, dict, list, or other types
    """
    # Process MCP result content, preserving structured data
    if hasattr(result, "content") and result.content is not None:
        try:
            content_data = result.content
            if debug:
                print(f"🔍 Content data type: {type(content_data)}")

            # Check if this is an ImageResult
            if isinstance(content_data, list) and len(content_data) > 0:
                # Check if the first item is text content with image data
                first_item = content_data[0]
                if hasattr(first_item, "text"):
                    try:
                        # Try to parse the text as JSON
                        text_data = json.loads(first_item.text)
                        if (
                            isinstance(text_data, dict)
                            and text_data.get("type") == "image_result"
                            and "image_base64" in text_data
                        ):
                            # Handle image result specially
                            if debug:
                                print(
                                    f"🖼️  Detected image result from {tool_name or 'tool'}"
                                )
                            return (
                                f"Image result from {tool_name or 'tool'} (base64 data)"
                            )
                    except (json.JSONDecodeError, AttributeError):
                        pass

                # Check if this is structured data (list of content items with structured data)
                # If all items are text content, extract as string
                # If any item contains structured data, preserve the structure
                has_structured_data = False
                text_parts = []
                structured_items = []

                for item in content_data:
                    if hasattr(item, "text"):
                        try:
                            # Try to parse as JSON to see if it's structured data
                            parsed = json.loads(item.text)
                            if (
                                isinstance(parsed, dict)
                                and "__structured_data__" in parsed
                            ):
                                # This is our special format for structured data
                                structured_items.append(parsed["__structured_data__"])
                                has_structured_data = True
                            elif isinstance(parsed, (dict, list)):
                                structured_items.append(parsed)
                                has_structured_data = True
                            else:
                                text_parts.append(item.text)
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, treat as text
                            text_parts.append(item.text)
                    elif isinstance(item, dict) and "text" in item:
                        try:
                            # Try to parse as JSON to see if it's structured data
                            parsed = json.loads(item["text"])
                            if (
                                isinstance(parsed, dict)
                                and "__structured_data__" in parsed
                            ):
                                # This is our special format for structured data
                                structured_items.append(parsed["__structured_data__"])
                                has_structured_data = True
                            elif isinstance(parsed, (dict, list)):
                                structured_items.append(parsed)
                                has_structured_data = True
                            else:
                                text_parts.append(item["text"])
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, treat as text
                            text_parts.append(item["text"])
                    else:
                        text_parts.append(str(item))

                if has_structured_data:
                    if debug:
                        print("📊 Preserving structured data from content list")
                    # If we have both text and structured data, combine them
                    if text_parts:
                        result_data: Any = {
                            "text": "".join(text_parts),
                            "structured_data": structured_items,
                        }
                    else:
                        # If we only have structured data, return the first item or combine them
                        if len(structured_items) == 1:
                            result_data = structured_items[0]
                        else:
                            result_data = structured_items
                    return result_data
                else:
                    # All text content
                    if debug:
                        print("📝 Converting content list to string")
                    return "".join(text_parts)
            elif (
                isinstance(content_data, dict)
                and content_data.get("type") == "image_result"
                and "image_base64" in content_data
            ):
                # Handle image result specially
                if debug:
                    print(f"🖼️  Detected image result from {tool_name or 'tool'}")
                return f"Image result from {tool_name or 'tool'} (base64 data)"
            else:
                # For other content types, preserve structure if it's dict/list
                if isinstance(content_data, dict):
                    # Check if this is our special format for structured data
                    if "__structured_data__" in content_data:
                        if debug:
                            print("📊 Extracting structured data from special format")
                        return content_data["__structured_data__"]
                    else:
                        if debug:
                            print("📊 Preserving structured data")
                        return content_data
                elif isinstance(content_data, list):
                    if debug:
                        print("📊 Preserving structured data")
                    return content_data
                else:
                    if debug:
                        print("📝 Converting content to string")
                    return str(content_data)
        except Exception as e:
            if debug:
                print(f"❌ Error processing content: {e}")
            return str(result.content)
    else:
        if debug:
            print("📝 Converting result to string")
        return str(result)


def run_async_in_sync_context(async_func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Run an async function in a synchronous context using nest_asyncio.

    This utility function handles the common pattern of running async code
    from within a synchronous context, which is needed for tool execution
    in evaluation tasks.

    Args:
        async_func: The async function to run
        *args: Positional arguments for the async function
        **kwargs: Keyword arguments for the async function

    Returns:
        The result of the async function

    Raises:
        Exception: Any exception raised by the async function
    """
    import asyncio
    import nest_asyncio

    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()

    # Create the async function call
    async def _call_async() -> Any:
        return await async_func(*args, **kwargs)

    # Run the async function
    return asyncio.run(_call_async())


# =============================================================================
# Shared helpers used by optimizer and evaluator
# =============================================================================


def init_opik_and_load_dataset(dataset_name: str, console: Console) -> tuple[Opik, Any]:
    """Initialize Opik client and load dataset by name with fallback to opik_optimizer.datasets."""
    client = Opik()
    console.print(f"📊 Loading dataset: {dataset_name}")

    try:
        dataset = client.get_dataset(name=dataset_name)
        console.print("✅ Dataset loaded successfully from Opik")
    except Exception as opik_error:
        console.print(f"⚠️  Dataset not found in Opik: {opik_error}")
        console.print("🔍 Checking opik_optimizer.datasets...")
        import opik_optimizer.datasets as optimizer_datasets

        if hasattr(optimizer_datasets, dataset_name):
            dataset_func = getattr(optimizer_datasets, dataset_name)
            console.print(
                f"📦 Creating dataset using opik_optimizer.datasets.{dataset_name}"
            )
            dataset = dataset_func()
            console.print("✅ Dataset created successfully from opik_optimizer")
        else:
            raise AttributeError(
                f"Dataset function '{dataset_name}' not found in opik_optimizer.datasets"
            )

    return client, dataset


def resolve_prompt_with_opik(
    client: Opik, prompt_value: str, console: Console
) -> tuple[str, Optional[str]]:
    """Resolve prompt by name via Opik, falling back to the provided value.

    Returns:
        tuple: (prompt_content, prompt_id) where prompt_id is None if not found in Opik
    """
    # First check if prompt_value is a filename and load content from file
    import os

    if os.path.isfile(prompt_value):
        try:
            console.print(f"📁 Loading prompt from file: {prompt_value}")
            with open(prompt_value, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
            if file_content:
                console.print(
                    f"✅ Loaded prompt from file: {file_content[:100]}{'...' if len(file_content) > 100 else ''}"
                )
                return file_content, None  # No prompt_id for file-based prompts
            else:
                console.print(
                    f"⚠️  File '{prompt_value}' is empty, falling back to Opik lookup"
                )
        except Exception as e:
            console.print(
                f"⚠️  Error reading file '{prompt_value}': {e}, falling back to Opik lookup"
            )

    # If not a file or file loading failed, try Opik lookup
    try:
        console.print(f"🔍 Looking up prompt '{prompt_value}' in Opik...")
        prompt = client.get_prompt(name=prompt_value)
        prompt_content = (
            prompt.prompt if prompt and hasattr(prompt, "prompt") else str(prompt)
        )
        if not prompt_content or prompt_content == "None":
            console.print(
                "⚠️  Prompt found in Opik but content is None/empty/'None', using original prompt value"
            )
            return prompt_value, None
        console.print(
            f"✅ Found prompt in Opik: {prompt_content[:100]}{'...' if len(prompt_content) > 100 else ''}"
        )
        # Extract prompt_id from the prompt object if available
        prompt_id = getattr(prompt, "id", None) or getattr(prompt, "prompt_id", None)
        if prompt_id:
            console.print(f"📋 Prompt ID: {prompt_id}")
        return prompt_content, prompt_id
    except Exception as e:
        console.print(
            f"⚠️  Prompt '{prompt_value}' not found in Opik ({e}), using as direct prompt"
        )
        return prompt_value, None


def _list_available_metrics_from_module(metrics_module: Any) -> List[str]:
    """List all callable metric names from a metrics module."""
    return sorted(
        [
            name
            for name in dir(metrics_module)
            if not name.startswith("_") and callable(getattr(metrics_module, name))
        ]
    )


def _load_metrics_from_file(file_path: str, console: Optional[Console] = None) -> Any:
    """Load a Python module from file to source custom metrics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metric file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("metric_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if console:
        console.print(f"✅ Loaded metrics from file: {file_path}")
    return module


def load_metrics_by_names(
    metric_names_csv: str, metrics_file: Optional[str], console: Console
) -> List[Any]:
    """Load metrics by name from either opik.evaluation.metrics OR --metrics-file.

    Metrics can be either:
    1. A class that when instantiated has a score() method requiring --output mapping
    2. A function that takes (dataset_item, llm_output) as parameters

    Returns a list of metric instances (class instances or function adapters).
    """
    names = [name.strip() for name in metric_names_csv.split(",")]
    metric_instances: List[Any] = []
    custom_module = (
        _load_metrics_from_file(metrics_file, console) if metrics_file else None
    )

    for metric_name in names:
        metric_obj = None
        source_module = None

        # Try custom module first if provided
        if custom_module and hasattr(custom_module, metric_name):
            metric_obj = getattr(custom_module, metric_name)
            source_module = "custom metrics file"
        else:
            # Try opik.evaluation.metrics
            if hasattr(opik_metrics, metric_name):
                metric_obj = getattr(opik_metrics, metric_name)
                source_module = "opik.evaluation.metrics"

        if metric_obj is None:
            console.print(f"❌ Unknown metric '{metric_name}'. Available metrics:")
            if custom_module:
                console.print("   From custom metrics file:")
                for available_metric in _list_available_metrics_from_module(
                    custom_module
                ):
                    console.print(f"     - {available_metric}")
            console.print("   From opik.evaluation.metrics:")
            for available_metric in _list_available_metrics_from_module(opik_metrics):
                console.print(f"     - {available_metric}")
            raise ValueError(f"Unknown metric: {metric_name}")

        # Check if it's a class or a function
        if isinstance(metric_obj, type):
            # It's a class - instantiate it
            metric_instance = metric_obj()

            # Ensure OPIK integration for ALL metrics (both custom and built-in)
            from opik.evaluation.metrics import BaseMetric

            if isinstance(metric_instance, BaseMetric):
                # Enable OPIK tracking for all metrics
                if hasattr(metric_instance, "track"):
                    metric_instance.track = True
                    if source_module == "custom metrics file":
                        console.print(
                            f"✅ Enabled OPIK tracking for custom metric: {metric_name}"
                        )
                    else:
                        console.print(
                            f"✅ Enabled OPIK tracking for built-in metric: {metric_name}"
                        )
            else:
                console.print(
                    f"⚠️  Warning: Metric '{metric_name}' should inherit from BaseMetric for proper OPIK integration"
                )

            metric_instances.append(metric_instance)
            console.print(
                f"✅ Loaded metric class: {metric_name} (from {source_module})"
            )

        elif callable(metric_obj):
            # It's a function - create an adapter class
            from opik.evaluation.metrics import BaseMetric

            class FunctionMetricAdapter(BaseMetric):
                """Adapter class to wrap a function metric for use with Opik evaluate()."""

                def __init__(self, metric_fn: Callable, metric_name: str):
                    super().__init__()
                    self.metric_fn = metric_fn
                    self.metric_name_str = metric_name

                def score(self, **kwargs) -> Any:
                    """Score method that adapts function signature to Opik's expected format.

                    Expects kwargs like 'output' (llm_output) and other fields.
                    Constructs dataset_item from kwargs (excluding 'output') and calls the function.
                    """
                    # Extract llm_output from 'output' parameter (standard Opik convention)
                    llm_output = kwargs.pop("output", None)
                    if llm_output is None:
                        raise ValueError(
                            f"Function metric '{self.metric_name_str}' requires 'output' parameter "
                            f"(which should be mapped from task output via scoring_key_mapping)"
                        )

                    # Everything else goes into dataset_item
                    dataset_item = kwargs

                    # Call the original function
                    return self.metric_fn(dataset_item, llm_output)

            metric_instance = FunctionMetricAdapter(metric_obj, metric_name)
            if hasattr(metric_instance, "track"):
                metric_instance.track = True

            metric_instances.append(metric_instance)
            console.print(
                f"✅ Loaded metric function (adapted): {metric_name} (from {source_module})"
            )
        else:
            raise ValueError(
                f"Metric '{metric_name}' must be either a class or a callable function"
            )

    return metric_instances


def load_metrics_by_names_for_optimizer(
    metric_names_csv: str,
    metrics_file: Optional[str],
    console: Console,
    output_ref: str = "reference",
    reference_field: str = "answer",
) -> List[Any]:
    """Load metrics by name from either opik.evaluation.metrics OR --metrics-file (for optimizer use).

    Metrics can be either:
    1. A class that when instantiated has a score() method requiring --output mapping
    2. A function that takes (dataset_item, llm_output) as parameters

    For class metrics, creates a wrapper function that adapts to optimizer's expected signature.
    For function metrics, uses them directly.

    Args:
        metric_names_csv: Comma-separated list of metric names
        metrics_file: Optional path to Python file containing metrics
        console: Console for output
        output_ref: The parameter name expected by the metric's score() method (default: "reference")
        reference_field: The dataset field name to use as reference (default: "answer")

    Returns:
        List of callable functions that take (dataset_item, llm_output) as parameters
    """
    names = [name.strip() for name in metric_names_csv.split(",")]
    metric_functions: List[Any] = []

    custom_module = (
        _load_metrics_from_file(metrics_file, console) if metrics_file else None
    )

    for metric_name in names:
        metric_obj = None
        source_module = None

        # Try custom module first if provided
        if custom_module and hasattr(custom_module, metric_name):
            metric_obj = getattr(custom_module, metric_name)
            source_module = "custom metrics file"
        else:
            # Try opik.evaluation.metrics
            if hasattr(opik_metrics, metric_name):
                metric_obj = getattr(opik_metrics, metric_name)
                source_module = "opik.evaluation.metrics"

        if metric_obj is None:
            console.print(f"❌ Unknown metric '{metric_name}'. Available metrics:")
            if custom_module:
                console.print("   From custom metrics file:")
                for available_metric in _list_available_metrics_from_module(
                    custom_module
                ):
                    console.print(f"     - {available_metric}")
            console.print("   From opik.evaluation.metrics:")
            for available_metric in _list_available_metrics_from_module(opik_metrics):
                console.print(f"     - {available_metric}")
            raise ValueError(f"Unknown metric: {metric_name}")

        # Check if it's a class or a function
        if isinstance(metric_obj, type):
            # It's a class - create a wrapper function
            metric_instance = metric_obj()

            # Create a wrapper function that adapts the class's score() method
            # to the optimizer's expected signature (dataset_item, llm_output)
            def make_class_metric_wrapper(metric_inst, metric_nm, ref_param, ref_field):
                def class_metric_wrapper(dataset_item, llm_output):
                    """Wrapper function that calls class metric's score() method."""
                    # Extract the reference value from dataset_item
                    reference_value = dataset_item.get(ref_field)

                    # Call the metric's score() method with the expected parameters
                    # The score() method typically expects 'output' (llm_output) and 'reference' (or other name)
                    try:
                        # Try calling with output and the reference parameter name
                        result = metric_inst.score(
                            output=llm_output, **{ref_param: reference_value}
                        )
                        return result
                    except Exception:
                        # If that fails, try with just output and reference (common case)
                        try:
                            result = metric_inst.score(
                                output=llm_output, reference=reference_value
                            )
                            return result
                        except Exception:
                            # If that also fails, try inspecting the signature
                            import inspect

                            sig = inspect.signature(metric_inst.score)
                            param_names = list(sig.parameters.keys())
                            param_names.remove("self")

                            # Build kwargs based on available parameters
                            kwargs = {"output": llm_output}
                            if ref_param in param_names:
                                kwargs[ref_param] = reference_value
                            elif "reference" in param_names:
                                kwargs["reference"] = reference_value
                            elif len(param_names) > 0:
                                # Use first parameter after output for reference
                                other_params = [p for p in param_names if p != "output"]
                                if other_params:
                                    kwargs[other_params[0]] = reference_value

                            result = metric_inst.score(**kwargs)
                            return result

                return class_metric_wrapper

            metric_fn = make_class_metric_wrapper(
                metric_instance, metric_name, output_ref, reference_field
            )
            metric_functions.append(metric_fn)
            console.print(
                f"✅ Loaded metric class (adapted): {metric_name} (from {source_module})"
            )

        elif callable(metric_obj):
            # It's a function - use it directly
            metric_functions.append(metric_obj)
            console.print(
                f"✅ Loaded metric function: {metric_name} (from {source_module})"
            )
        else:
            raise ValueError(
                f"Metric '{metric_name}' must be either a class or a callable function"
            )

    return metric_functions


# Global set to track temporary files for cleanup
_temp_files: set = set()


def _cleanup_temp_files() -> None:
    """Clean up all temporary files created during URL downloads."""
    for temp_path in _temp_files.copy():
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass  # Ignore cleanup errors
    _temp_files.clear()


# Register cleanup function to run at exit
atexit.register(_cleanup_temp_files)


def download_file_from_url(url: str, console: Optional[Console] = None) -> str:
    """
    Download a file from a URL and save it to a temporary file.

    Args:
        url: URL to download the file from
        console: Optional Rich console for output

    Returns:
        Path to the temporary file containing the downloaded content

    Raises:
        ValueError: If the URL is invalid or download fails
    """
    try:
        # Parse the URL to validate it
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}")

        if console:
            console.print(f"🌐 Downloading file from URL: {url}")
        else:
            print(f"🌐 Downloading file from URL: {url}")

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Track the temporary file for cleanup
        _temp_files.add(temp_path)

        # Download the file content
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")

        # Write content to temporary file
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)

        if console:
            console.print(f"✅ Downloaded file to temporary location: {temp_path}")
        else:
            print(f"✅ Downloaded file to temporary location: {temp_path}")

        return temp_path

    except urllib.error.URLError as e:
        error_msg = f"Failed to download file from URL {url}: {e}"
        if console:
            console.print(f"❌ {error_msg}")
        else:
            print(f"❌ {error_msg}")
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error downloading file from URL {url}: {e}"
        if console:
            console.print(f"❌ {error_msg}")
        else:
            print(f"❌ {error_msg}")
        raise ValueError(error_msg)


def resolve_tools_file_path(tools_file: str, console: Optional[Console] = None) -> str:
    """
    Resolve a tools file path, downloading from URL if necessary.

    Args:
        tools_file: File path or URL to the tools file
        console: Optional Rich console for output

    Returns:
        Resolved file path (local file path or temporary file path for URLs)
    """
    # Check if it's a URL
    if tools_file.startswith(("http://", "https://")):
        return download_file_from_url(tools_file, console)
    else:
        # It's a local file path
        return tools_file


@track
async def chat_with_tools(
    user_text: str,
    system_prompt: str,
    model: str,
    model_parameters: Dict[str, Any],
    mcp_manager: Any,
    messages: List[Dict[str, Any]],
    max_rounds: int = 4,
    debug: bool = False,
    console: Optional[Console] = None,
    thread_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
) -> str:
    """
    Shared chat function that handles LLM calls with tool execution.

    This function provides the robust chat logic from chatbot.py that can be
    reused by both the chatbot and evaluator.

    Args:
        user_text: The user's input text
        system_prompt: The system prompt to use
        model: The LLM model to use
        model_parameters: Additional model parameters
        mcp_manager: The MCP manager instance
        messages: List of messages for conversation history
        max_rounds: Maximum number of conversation rounds
        debug: Whether to enable debug output
        console: Optional Rich console for output
        thread_id: Optional thread ID for Opik tracing

    Returns:
        The final response text
    """
    if not mcp_manager.sessions:
        raise RuntimeError("Not connected to any MCP servers.")

    # Generate mermaid diagram and update Opik context with metadata
    mermaid_diagram = await generate_mcp_mermaid_diagram(mcp_manager)

    try:
        from opik import opik_context

        context_updates: Dict[str, Any] = {}
        if thread_id:
            context_updates["thread_id"] = thread_id
        if prompt_id:
            context_updates["prompt_id"] = prompt_id
        if mermaid_diagram:
            # Get existing metadata and merge, preserving what's already there
            existing_metadata = {}
            try:
                current_trace = opik_context.get_current_trace()
                if (
                    current_trace
                    and hasattr(current_trace, "metadata")
                    and current_trace.metadata
                ):
                    existing_metadata = (
                        current_trace.metadata.copy()
                        if isinstance(current_trace.metadata, dict)
                        else {}
                    )
            except Exception:
                pass

            # Set _opik_graph_definition within metadata
            existing_metadata["_opik_graph_definition"] = {
                "format": "mermaid",
                "data": mermaid_diagram,
            }
            context_updates["metadata"] = existing_metadata

        if context_updates:
            opik_context.update_current_trace(**context_updates)
    except Exception:
        # Opik not available, continue without tracing
        pass

    # 1) Fetch tool catalog from all MCP servers
    tools = await mcp_manager._get_all_tools()

    # 2) Add user message to persistent history
    user_msg = {"role": "user", "content": user_text}
    messages.append(user_msg)

    # 3) Chat loop with tool calling using persistent messages
    text_reply: str = ""

    for round_num in range(max_rounds):
        try:
            if debug:
                print(f"🔄 LLM call round {round_num + 1}/{max_rounds}")

            # Show spinner while processing (if console available)
            if console:
                with console.status(
                    "[bold green]Thinking...[/bold green]", spinner="dots"
                ):
                    # Call LLM with proper span management within the current trace
                    resp = call_llm_with_tracing(
                        model=model,
                        messages=messages,
                        tools=tools if tools else None,
                        debug=debug,
                        console=console,
                        **model_parameters,
                    )
            else:
                # Call LLM without spinner
                resp = call_llm_with_tracing(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    debug=debug,
                    console=console,
                    **model_parameters,
                )

            # Use common utility function to extract content and tool calls
            content, tool_calls = extract_llm_content(resp, debug)

            if not tool_calls:
                text_reply = (content or "").strip()
                # Add assistant's final response to persistent history
                messages.append({"role": "assistant", "content": text_reply})
                break
        except Exception as e:
            if debug:
                print(f"❌ LLM call failed in round {round_num + 1}: {e}")
            text_reply = f"Error in LLM call: {e}"
            break

        # 4) Execute each requested tool via MCP
        executed_tool_msgs: List[Dict[str, Any]] = []
        assistant_tool_stub = []

        for tc in tool_calls:
            # Show spinner while executing tool (if console available)
            if console:
                with console.status(
                    f"[bold blue]Executing {tc.function.name}...[/bold blue]",
                    spinner="dots",
                ):
                    tool_result = await mcp_manager.execute_tool_call(tc)
            else:
                tool_result = await mcp_manager.execute_tool_call(tc)

            # Build messages to feed back to the model
            assistant_tool_stub.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )
            executed_tool_msgs.append(format_tool_result(tc.id, tool_result))

        # Add the assistant tool-call stub + tool results to persistent history
        messages.append(format_assistant_tool_calls(assistant_tool_stub))
        messages.extend(executed_tool_msgs)

    return text_reply
