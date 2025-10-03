# python client.py [config.json]
# Example: python client.py config.json
# If no config file is provided, uses config.json by default

import asyncio
import sys
import json
import os
import subprocess
import uuid
import argparse
import base64
import tempfile
import io
import traceback
from contextlib import AsyncExitStack
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from PIL import Image

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from litellm import completion  # can handle tools
from litellm.integrations.opik.opik import OpikLogger
import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.status import Status
from opik import track, opik_context
import opik

# Prompt toolkit imports for enhanced input handling
from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion, WordCompleter
from prompt_toolkit.completion import merge_completers
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.shortcuts import confirm

load_dotenv()


def configure_opik(opik_mode: str = "hosted"):
    """Configure Opik based on the specified mode."""
    if opik_mode == "disabled":
        return

    # Set the project name via environment variable
    os.environ["OPIK_PROJECT_NAME"] = "ez-mcp-chatbot"

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




@dataclass
class ServerConfig:
    name: str
    description: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


def _mcp_tools_to_openai_tools(tools_resp) -> List[Dict[str, Any]]:
    # Map MCP tool spec to OpenAI function tools
    tools = []
    for t in tools_resp.tools:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    # MCP provides a proper JSON schema in inputSchema
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _mk_user_msg(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def _mk_assistant_tool_msg(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Record the assistant's tool calls so the model has the chain
    return {"role": "assistant", "tool_calls": tool_calls, "content": ""}


def _mk_tool_result_msg(tool_call_id: str, content: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


class ChatbotCompleter(Completer):
    """Custom completer for the chatbot with command and Python code completion."""

    def __init__(self):
        # Basic commands
        self.commands = ["/clear", "quit", "exit", "help"]

        # Python built-ins and common functions for ! commands
        self.python_keywords = [
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "tuple",
            "set",
            "bool",
            "type",
            "isinstance",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "sorted",
            "reversed",
            "sum",
            "min",
            "max",
            "abs",
            "round",
            "divmod",
            "pow",
            "bin",
            "hex",
            "oct",
            "chr",
            "ord",
            "open",
            "input",
            "raw_input",
            "file",
            "dir",
            "vars",
            "locals",
            "globals",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "callable",
            "issubclass",
            "super",
            "property",
            "staticmethod",
            "classmethod",
            "all",
            "any",
            "ascii",
            "bytearray",
            "bytes",
            "complex",
            "frozenset",
            "memoryview",
            "object",
            "slice",
            "None",
            "True",
            "False",
            "Ellipsis",
            "NotImplemented",
            "import",
            "from",
            "as",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "def",
            "class",
            "return",
            "yield",
            "break",
            "continue",
            "pass",
            "del",
            "global",
            "nonlocal",
            "lambda",
            "and",
            "or",
            "not",
            "in",
            "is",
            "assert",
            "raise",
            "exec",
            "eval",
            "self",
        ]

        # Chatbot-specific attributes that users might want to access
        self.chatbot_attributes = [
            "self.sessions",
            "self.model",
            "self.messages",
            "self.console",
            "self.thread_id",
            "self.servers",
            "self.max_rounds",
            "self.get_messages()",
            "self.get_message_count()",
            "self.clear_messages()",
            # Tool execution helpers
            "self.run_tool()",
            "self.list_available_tools()",
            "self.call_session_tool()",
            # Private methods
            "self._setup_prompt_toolkit()",
            "self._connect_server()",
            "self._get_all_tools()",
            "self._execute_tool_call()",
            "self._handle_image_result()",
            "self._execute_python_code()",
            "self._call_llm_with_span()",
            # Useful private method calls for exploration
            "self._get_all_tools()",
            "self._execute_tool_call()",
            "self._handle_image_result()",
            # Example calls with parameters (for reference)
            'self._execute_python_code("print(42)")',
            'self._handle_image_result({}, "test")',
            # Tool execution examples
            'self.run_tool("tool_name", param1="value")',
            "self.list_available_tools()",
            'self.call_session_tool("server_name", "tool_name", param="value")',
            # Tool execution
            'run_tool("server_name", "tool_name", param="value")',
            'run_tool_return("server_name", "tool_name", param="value")',
            # Sync tool helpers
            "get_tools()",
            'get_tools("server_name")',
        ]

    def get_completions(self, document, complete_event):
        """Provide completions based on the current input."""
        text = document.text
        word = document.get_word_before_cursor()

        # Command completions (for commands starting with / or basic commands)
        if text.startswith("/") or text in ["quit", "exit", "help"]:
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))

        # Python code completions (for commands starting with !)
        elif text.startswith("!"):
            python_text = text[1:]  # Remove the ! prefix
            python_word = python_text.split()[-1] if python_text.split() else ""

            # First, try chatbot-specific attributes
            for attr in self.chatbot_attributes:
                if attr.startswith(python_word):
                    # Don't add extra ! prefix since text already starts with !
                    yield Completion(
                        f"{python_text.replace(python_word, attr)}",
                        start_position=-len(python_word),
                    )

            # Then try Python keywords
            for keyword in self.python_keywords:
                if keyword.startswith(python_word):
                    # Don't add extra ! prefix since text already starts with !
                    yield Completion(
                        f"{python_text.replace(python_word, keyword)}",
                        start_position=-len(python_word),
                    )

        # General word completions for other cases
        else:
            # Could add more sophisticated completion here
            pass


class MCPChatbot:
    def __init__(
        self,
        config_path: str,
        system_prompt: str,
        max_rounds: Optional[int] = 4,
        debug: bool = False,
    ):
        self.system_prompt = system_prompt
        self.servers, self.model, self.model_kwargs = self.load_config(config_path)
        self.max_rounds = max_rounds
        self.debug = debug
        self.sessions: Dict[str, ClientSession] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.exit_stack = AsyncExitStack()
        self.console = Console()
        # Generate unique thread-id for this chatbot instance
        self.thread_id = str(uuid.uuid4())
        self.clear_messages()

        # Set up prompt_toolkit for enhanced input handling
        self._setup_prompt_toolkit()

        # Set up persistent Python evaluation environment
        self._setup_python_environment()

    def _setup_prompt_toolkit(self):
        """Set up prompt_toolkit for enhanced input handling with history and completion."""
        # Set up history file
        history_file = os.path.expanduser("~/.opik_mcp_chatbot_history")

        # Create prompt session with history and completion
        self.prompt_session = PromptSession(
            history=FileHistory(history_file),
            completer=ChatbotCompleter(),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
        )

    def _setup_python_environment(self):
        """Set up persistent Python evaluation environment."""
        # Create persistent execution environment
        self.python_globals = {
            # Include all built-ins and modules
            "__builtins__": __builtins__,
            # Make the chatbot instance available as 'self'
            "self": self,
            # Add async support
            "asyncio": __import__("asyncio"),
            "await": self._create_await_helper(),
            # Add tool execution helpers
            "run_tool": self._create_direct_tool_runner(),
            "run_tool_return": self._create_direct_tool_runner_return(),
            # Add sync tool helpers
            "get_tools": self._create_direct_tool_getter(),
            "get_tool_info": self._create_direct_tool_info_getter(),
        }

        # Initialize with some useful imports
        exec("import json, os, sys, traceback", self.python_globals)
        exec("from datetime import datetime", self.python_globals)

    @staticmethod
    def load_config(
        config_path: str = "config.json",
    ) -> tuple[List[ServerConfig], str, Dict[str, Any]]:
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Use default configuration when no config file exists
            config = {
                "model": "openai/gpt-4o-mini",
                "model_kwargs": {"temperature": 0.2},
                "mcp_servers": [
                    {
                        "name": "ez-mcp-server",
                        "description": "Ez MCP server with default tools",
                        "command": "ez-mcp-server",
                        "args": [],
                    }
                ],
            }

        servers = []
        for server_data in config.get("mcp_servers", []):
            # Expand environment variables in env dict
            env = server_data.get("env", {})
            expanded_env = {}
            for key, value in env.items():
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    expanded_env[key] = os.getenv(env_var, "")
                else:
                    expanded_env[key] = value

            servers.append(
                ServerConfig(
                    name=server_data["name"],
                    description=server_data.get("description", ""),
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=expanded_env if expanded_env else None,
                )
            )

        # Extract model configuration
        model = config.get("model", "openai/gpt-4o-mini")
        model_kwargs = config.get("model_kwargs", {"temperature": 0.2})

        return servers, model, model_kwargs

    async def connect_all_servers(self):
        """Connect to all configured MCP servers via subprocess."""
        for server_config in self.servers:
            try:
                await self._connect_server(server_config)
                self.console.print(
                    f"[green]✓[/green] Connected to [bold]{server_config.name}[/bold]: {server_config.description}"
                )
            except Exception as e:
                self.console.print(
                    f"[red]✗[/red] Failed to connect to [bold]{server_config.name}[/bold]: {e}"
                )

    async def _connect_server(self, server_config: ServerConfig):
        """Connect to a single MCP server via subprocess."""
        # Set up environment variables for the subprocess
        if server_config.env:
            # Update the current process environment for the subprocess
            original_env = {}
            for key, value in server_config.env.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            # Create MCP client session using stdio client
            params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
            )

            transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdin, write = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdin, write)
            )
            await session.initialize()

            self.sessions[server_config.name] = session
        finally:
            # Restore original environment variables
            if server_config.env:
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value

    async def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Aggregate tools from all connected MCP servers."""
        all_tools = []
        for server_name, session in self.sessions.items():
            try:
                tools_resp = await session.list_tools()
                server_tools = _mcp_tools_to_openai_tools(tools_resp)
                # Prefix tool names with server name to avoid conflicts
                for tool in server_tools:
                    tool["function"][
                        "name"
                    ] = f"{server_name}_{tool['function']['name']}"
                all_tools.extend(server_tools)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning:[/yellow] Failed to get tools from [bold]{server_name}[/bold]: {e}"
                )
        return all_tools

    @track(name="execute_tool_call", type="tool")
    async def _execute_tool_call(self, tool_call) -> str:
        """Execute a tool call on the appropriate MCP server."""
        fn_name = tool_call.function.name
        args_raw = tool_call.function.arguments or "{}"
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}

        # Parse server name from tool name (format: server_name_tool_name)
        if "_" in fn_name:
            # Find the first underscore to split server name from tool name
            parts = fn_name.split("_", 1)
            if len(parts) == 2:
                server_name, actual_tool_name = parts
            else:
                # Fallback: treat as tool name without server prefix
                server_name = None
                actual_tool_name = fn_name
        else:
            # Fallback: try to find the tool in any server
            server_name = None
            actual_tool_name = fn_name

        if server_name and server_name in self.sessions:
            session = self.sessions[server_name]
        else:
            # Try to find the tool in any connected server
            session = None
            for srv_name, sess in self.sessions.items():
                try:
                    tools_resp = await sess.list_tools()
                    tool_names = [t.name for t in tools_resp.tools]
                    if actual_tool_name in tool_names:
                        session = sess
                        break
                except Exception:
                    continue

            if session is None:
                return f"Error: Tool '{actual_tool_name}' not found in any connected server"

        try:
            # Log tool call start with input details
            if self.debug:
                print(f"🔧 Calling tool: {actual_tool_name} with args: {args}")

            # Call the MCP tool
            result = await session.call_tool(actual_tool_name, args)

            # Best-effort stringify of MCP result content
            if hasattr(result, "content") and result.content is not None:
                try:
                    content_data = result.content
                    if self.debug:
                        print(f"🔍 Content data type: {type(content_data)}")
                        print(
                            f"🔍 Content data keys: {content_data.keys() if isinstance(content_data, dict) else 'Not a dict'}"
                        )

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
                                    if self.debug:
                                        print(
                                            f"🖼️  Detected image result from {actual_tool_name}"
                                        )
                                    return self._handle_image_result(
                                        text_data, actual_tool_name
                                    )
                            except (json.JSONDecodeError, AttributeError):
                                pass
                        # If not an image, convert to string
                        if self.debug:
                            print(f"📝 Converting content to string")
                        content_str = str(content_data)
                    elif (
                        isinstance(content_data, dict)
                        and content_data.get("type") == "image_result"
                        and "image_base64" in content_data
                    ):
                        # Handle image result specially
                        if self.debug:
                            print(f"🖼️  Detected image result from {actual_tool_name}")
                        return self._handle_image_result(content_data, actual_tool_name)
                    else:
                        if self.debug:
                            print(f"📝 Converting content to JSON string")
                        content_str = json.dumps(content_data, separators=(",", ":"))
                except Exception as e:
                    if self.debug:
                        print(f"❌ Error processing content: {e}")
                    content_str = str(result.content)
            else:
                if self.debug:
                    print(f"📝 Converting result to string")
                content_str = str(result)

            # Log tool call result
            if self.debug:
                print(f"✅ Tool {actual_tool_name} completed successfully")
                print(f"📊 Result length: {len(content_str)} characters")

            # Check for excessively large results
            max_result_size = 10 * 1024 * 1024  # 10MB limit
            if len(content_str) > max_result_size:
                if self.debug:
                    print(
                        f"⚠️  Large result detected: {len(content_str):,} characters (limit: {max_result_size:,})"
                    )
                return (
                    f"⚠️ **Large result from {actual_tool_name}**\n\n"
                    f"📊 **Result Info:**\n"
                    f"- Size: {len(content_str):,} characters\n"
                    f"- Tool: {actual_tool_name}\n"
                    f"- Status: Completed successfully\n\n"
                    f"*Note: Result too large to display inline. Consider using a different approach or tool.*"
                )

            # The @track decorator will automatically capture:
            # - Function name (actual_tool_name)
            # - Input arguments (args)
            # - Output result (content_str)
            # - Execution time
            # - Success/failure status

            return content_str
        except Exception as e:
            if self.debug:
                print(f"❌ Tool {actual_tool_name} failed: {e}")
            # The @track decorator will capture the error details
            return f"Error executing tool '{actual_tool_name}': {e}"

    def _handle_image_result(self, image_data: Dict[str, Any], tool_name: str) -> str:
        """Handle image results by saving to temporary file and returning file reference."""
        try:
            # Decode base64 image data
            image_base64 = image_data.get("image_base64", "")
            content_type = image_data.get("content_type", "image/png")

            if not image_base64:
                return f"Error: No image data received from {tool_name}"

            # Decode base64 to bytes to get image info
            image_bytes = base64.b64decode(image_base64)

            # Create PIL Image from bytes to get dimensions
            image = Image.open(io.BytesIO(image_bytes))

            # Resize image to 400x300 pixels to keep them small for display
            resized_image = image.resize((400, 300), Image.Resampling.LANCZOS)

            # Convert back to bytes for saving
            output_buffer = io.BytesIO()
            resized_image.save(output_buffer, format="PNG")
            image_bytes = output_buffer.getvalue()

            # Update the image object to the resized version
            image = resized_image

            # Determine file extension from content type
            file_ext = ".png"  # default
            if "jpeg" in content_type or "jpg" in content_type:
                file_ext = ".jpg"
            elif "gif" in content_type:
                file_ext = ".gif"
            elif "bmp" in content_type:
                file_ext = ".bmp"
            elif "tiff" in content_type:
                file_ext = ".tiff"

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            # Log the image creation
            if self.debug:
                print(f"🖼️ Image created by {tool_name}")
                print(f"📊 Image size: {len(image_bytes)} bytes")
                print(
                    f"📐 Image dimensions: {image.size[0]}x{image.size[1]} pixels (resized to 400x300)"
                )
                print(f"💾 Image saved to: {tmp_path}")

            # Clean up the path for display (remove sandbox: prefix if present)
            if self.debug:
                print(f"🔍 Original tmp_path: {tmp_path}")
            display_path = tmp_path
            if tmp_path.startswith("sandbox:"):
                display_path = tmp_path[8:]  # Remove "sandbox:" prefix
                if self.debug:
                    print(f"🔍 Cleaned display_path: {display_path}")
            else:
                if self.debug:
                    print(f"🔍 No sandbox prefix found, using original: {display_path}")

            # Try to display image directly in console if supported
            try:
                import subprocess
                import shutil

                # Check if we're in a terminal that supports image display
                if shutil.which("kitty"):
                    # Kitty supports inline images
                    if self.debug:
                        print(f"🖼️ Displaying image in terminal...")
                    subprocess.run(
                        ["kitty", "+kitten", "icat", display_path], check=False
                    )
                    image_displayed = True
                elif shutil.which("wezterm"):
                    # WezTerm supports images
                    if self.debug:
                        print(f"🖼️ Displaying image in terminal...")
                    subprocess.run(["wezterm", "imgcat", display_path], check=False)
                    image_displayed = True
                else:
                    image_displayed = False
            except Exception:
                image_displayed = False

            if image_displayed:
                return (
                    f"🖼️ **Image created by {tool_name}**\n\n"
                    f"📊 **Image Details:**\n"
                    f"- Size: {len(image_bytes):,} bytes\n"
                    f"- Dimensions: {image.size[0]}x{image.size[1]} pixels\n"
                    f"- Format: {content_type}\n"
                    f"- Saved to: `{display_path}`"
                )
            else:
                # Simple fallback - just show the image info
                return (
                    f"🖼️ **Image created by {tool_name}**\n\n"
                    f"📊 **Image Details:**\n"
                    f"- Size: {len(image_bytes):,} bytes\n"
                    f"- Dimensions: {image.size[0]}x{image.size[1]} pixels\n"
                    f"- Format: {content_type}\n"
                    f"- Saved to: `{display_path}`\n\n"
                    f"💡 **Image saved successfully** - Click [Plot]({display_path}) to view"
                )

        except Exception as e:
            if self.debug:
                print(f"❌ Failed to handle image result from {tool_name}: {e}")
            return f"Error processing image from {tool_name}: {e}"

    def _execute_python_code(self, code: str) -> str:
        """Execute Python code with persistent environment."""
        try:
            # Use the persistent execution environment
            exec_globals = self.python_globals

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                # Try to evaluate as an expression first (single expressions only)
                try:
                    # Check if it's a simple expression (no semicolons, no colons for control flow)
                    if (
                        ";" not in code
                        and ":" not in code
                        and not code.strip().startswith(
                            (
                                "import ",
                                "from ",
                                "def ",
                                "class ",
                                "if ",
                                "for ",
                                "while ",
                                "try:",
                                "with ",
                            )
                        )
                    ):
                        result = eval(code, exec_globals)
                        output = captured_output.getvalue()

                        # If there's stdout output, show it
                        if output.strip():
                            if result is not None:
                                return f"🐍 Python Output:\n{output.strip()}\n🐍 Result: {repr(result)}"
                            else:
                                return f"🐍 Python Output:\n{output.strip()}"
                        else:
                            if result is not None:
                                return f"🐍 Result: {repr(result)}"
                            else:
                                return (
                                    "🐍 Python code executed successfully (no output)"
                                )
                    else:
                        raise SyntaxError("Not a simple expression")

                except (SyntaxError, NameError):
                    # If it's not a valid expression, try executing as a statement
                    # For multi-statement code, try to capture the last expression result
                    if ";" in code:
                        # Split by semicolon and try to evaluate the last part as an expression
                        parts = code.split(";")
                        # Execute all parts except the last
                        for part in parts[:-1]:
                            if part.strip():
                                exec(part.strip(), exec_globals)
                        # Try to evaluate the last part as an expression
                        try:
                            result = eval(parts[-1].strip(), exec_globals)
                            output = captured_output.getvalue()
                            if output.strip():
                                if result is not None:
                                    return f"🐍 Python Output:\n{output.strip()}\n🐍 Result: {repr(result)}"
                                else:
                                    return f"🐍 Python Output:\n{output.strip()}"
                            else:
                                if result is not None:
                                    return f"🐍 Result: {repr(result)}"
                                else:
                                    return "🐍 Python code executed successfully (no output)"
                        except:
                            # If last part isn't an expression, execute it too
                            exec(parts[-1].strip(), exec_globals)
                            output = captured_output.getvalue()
                            if output.strip():
                                return f"🐍 Python Output:\n{output.strip()}"
                            else:
                                return (
                                    "🐍 Python code executed successfully (no output)"
                                )
                    else:
                        # Single statement execution
                        exec(code, exec_globals)
                        output = captured_output.getvalue()

                        # If there's output, return it
                        if output.strip():
                            return f"🐍 Python Output:\n{output.strip()}"
                        else:
                            return "🐍 Python code executed successfully (no output)"

            finally:
                # Restore stdout
                sys.stdout = old_stdout

        except Exception as e:
            # Get the full traceback for better error reporting
            error_msg = traceback.format_exc()
            return f"🐍 Python Error:\n{error_msg}"

    def _create_await_helper(self):
        """Create a helper function to run async code from sync context."""

        def await_helper(coro):
            """Helper to run async code from sync context."""
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create a task
                task = loop.create_task(coro)
                # Return the task - user can access result with .result() if needed
                return task
            except RuntimeError:
                # No event loop, we can use asyncio.run
                return asyncio.run(coro)

        return await_helper

    def _create_direct_tool_runner(self):
        """Create a helper function to run tools via MCP sessions."""

        def run_tool(tool_identifier: str, **kwargs):
            """Helper to run tools - accepts SERVER.TOOL format.

            Args:
                tool_identifier: "SERVER.TOOL" format
                **kwargs: Arguments to pass to the tool
            """
            try:
                # Parse tool_identifier to extract server and tool names
                if "." not in tool_identifier:
                    return f"Error: Tool identifier must be in SERVER.TOOL format, got: {tool_identifier}"

                server_name, tool_name = tool_identifier.split(".", 1)

                # Check if server exists
                if server_name not in self.sessions:
                    available_servers = list(self.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to call the tool
                async def _call_tool():
                    session = self.sessions[server_name]
                    try:
                        result = await session.call_tool(tool_name, kwargs)

                        # Convert MCP result to readable string
                        if hasattr(result, "content") and result.content is not None:
                            try:
                                content_data = result.content
                                if (
                                    isinstance(content_data, list)
                                    and len(content_data) > 0
                                ):
                                    if content_data[0].get("type") == "text":
                                        return content_data[0]["text"]
                                    else:
                                        return str(content_data)
                                else:
                                    return str(content_data)
                            except Exception:
                                return str(result.content)
                        else:
                            return str(result)
                    except Exception as e:
                        return f"Error calling tool '{tool_name}': {e}"

                # Run the async function - use nest_asyncio to handle nested event loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    result = asyncio.run(_call_tool())
                    if self.debug:
                        print(result)
                    return None
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_call_tool())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result()
                        if self.debug:
                            print(result)
                        return None

            except Exception as e:
                return f"Error executing tool '{tool_identifier}': {e}"

        return run_tool

    def _create_direct_tool_runner_return(self):
        """Create a helper function to run tools via MCP sessions that returns the result."""

        def run_tool_return(tool_identifier: str, **kwargs):
            """Helper to run tools - accepts SERVER.TOOL format and returns the result.

            Args:
                tool_identifier: "SERVER.TOOL" format
                **kwargs: Arguments to pass to the tool

            Returns:
                The result from the tool execution
            """
            try:
                # Parse tool_identifier to extract server and tool names
                if "." not in tool_identifier:
                    return f"Error: Tool identifier must be in SERVER.TOOL format, got: {tool_identifier}"

                server_name, tool_name = tool_identifier.split(".", 1)

                # Check if server exists
                if server_name not in self.sessions:
                    available_servers = list(self.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to call the tool
                async def _call_tool():
                    session = self.sessions[server_name]
                    try:
                        result = await session.call_tool(tool_name, kwargs)

                        # Convert MCP result to readable string
                        if hasattr(result, "content") and result.content is not None:
                            try:
                                content_data = result.content
                                if (
                                    isinstance(content_data, list)
                                    and len(content_data) > 0
                                ):
                                    if content_data[0].get("type") == "text":
                                        return content_data[0]["text"]
                                    else:
                                        return str(content_data)
                                else:
                                    return str(content_data)
                            except Exception:
                                return str(result.content)
                        else:
                            return str(result)
                    except Exception as e:
                        return f"Error calling tool '{tool_name}': {e}"

                # Run the async function - use nest_asyncio to handle nested event loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    result = asyncio.run(_call_tool())
                    return result
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_call_tool())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result()
                        return result

            except Exception as e:
                return f"Error executing tool '{tool_identifier}': {e}"

        return run_tool_return

    def _create_direct_tool_getter(self):
        """Create a helper function to get tools via MCP sessions."""

        def get_tools(server_name: str = None):
            """Helper to get tools - prints rich formatted list of tools."""
            try:
                # If no server specified, show all servers and their tools
                if server_name is None:
                    if not self.sessions:
                        self.console.print(
                            "[bold red]No MCP servers connected[/bold red]"
                        )
                        self.console.print(
                            "Use `!list_available_tools()` to see connection status."
                        )
                        return None

                    # Create async function to get all tools for all servers
                    async def _get_all_tools_data():
                        all_data = []
                        for srv_name in self.sessions.keys():
                            try:
                                session = self.sessions[srv_name]
                                tools_resp = await session.list_tools()

                                server_data = {
                                    "name": srv_name,
                                    "tools": [],
                                    "error": None,
                                }

                                if tools_resp.tools:
                                    for tool in tools_resp.tools:
                                        server_data["tools"].append(
                                            {
                                                "name": tool.name,
                                                "description": tool.description
                                                or "No description available",
                                            }
                                        )

                                all_data.append(server_data)
                            except Exception as e:
                                all_data.append(
                                    {"name": srv_name, "tools": [], "error": str(e)}
                                )

                        return all_data

                    # Run the async function and then print
                    try:
                        import nest_asyncio

                        nest_asyncio.apply()
                        data = asyncio.run(_get_all_tools_data())
                    except Exception:
                        # Fallback: try to run in a new thread with new event loop
                        import concurrent.futures

                        def run_in_thread():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(
                                    _get_all_tools_data()
                                )
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_in_thread)
                            data = future.result()

                    # Now print the formatted output
                    self.console.print(
                        "[bold blue]# Available MCP Servers and Tools[/bold blue]"
                    )
                    self.console.print()

                    for server_data in data:
                        if server_data["error"]:
                            self.console.print(
                                f"[bold cyan]## 📡 Server: {server_data['name']}[/bold cyan]"
                            )
                            self.console.print()
                            self.console.print(
                                f"[italic red]*Error getting tools: {server_data['error']}*[/italic red]"
                            )
                            self.console.print()
                        else:
                            self.console.print(
                                f"[bold cyan]## 📡 Server: {server_data['name']}[/bold cyan]"
                            )
                            self.console.print()

                            if server_data["tools"]:
                                for tool in server_data["tools"]:
                                    self.console.print(
                                        f"[green]-[/green] [bold]{tool['name']}[/bold]: {tool['description']}"
                                    )
                            else:
                                self.console.print(
                                    "[italic]*No tools available*[/italic]"
                                )

                            self.console.print()

                    return None

                # Check if server exists
                if server_name not in self.sessions:
                    available_servers = list(self.sessions.keys())
                    self.console.print("[bold red]## Error[/bold red]")
                    self.console.print(
                        f"Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"
                    )
                    return None

                # Create async function to get tools for specific server
                async def _get_tools_data():
                    session = self.sessions[server_name]
                    try:
                        tools_resp = await session.list_tools()

                        tools_data = []
                        if tools_resp.tools:
                            for tool in tools_resp.tools:
                                tools_data.append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description
                                        or "No description available",
                                    }
                                )

                        return tools_data
                    except Exception as e:
                        raise e

                # Run the async function and then print
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    tools_data = asyncio.run(_get_tools_data())
                except Exception as e:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_get_tools_data())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        tools_data = future.result()

                # Now print the formatted output
                self.console.print(
                    f"[bold blue]# Tools for Server: {server_name}[/bold blue]"
                )
                self.console.print()

                if tools_data:
                    for tool in tools_data:
                        self.console.print(
                            f"[green]-[/green] [bold]{tool['name']}[/bold]: {tool['description']}"
                        )
                else:
                    self.console.print("[italic]*No tools available*[/italic]")

                return None

            except Exception as e:
                self.console.print("[bold red]## Error[/bold red]")
                self.console.print(f"Error getting tools: {e}")
                return None

        return get_tools

    def _create_direct_tool_info_getter(self):
        """Create a helper function to get detailed tool information via MCP sessions."""

        def get_tool_info(tool_identifier: str):
            """Helper to get detailed information about a specific tool.

            Args:
                tool_identifier: "SERVER.TOOL" format

            Returns:
                Detailed information about the tool including description, parameters, etc.
            """
            try:
                # Parse tool_identifier to extract server and tool names
                if "." not in tool_identifier:
                    return f"Error: Tool identifier must be in SERVER.TOOL format, got: {tool_identifier}"

                server_name, tool_name = tool_identifier.split(".", 1)

                # Check if server exists
                if server_name not in self.sessions:
                    available_servers = list(self.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to get tool info
                async def _get_tool_info():
                    session = self.sessions[server_name]
                    try:
                        tools_resp = await session.list_tools()

                        # Find the specific tool
                        target_tool = None
                        for tool in tools_resp.tools:
                            if tool.name == tool_name:
                                target_tool = tool
                                break

                        if not target_tool:
                            available_tools = [tool.name for tool in tools_resp.tools]
                            return f"Error: Tool '{tool_name}' not found in server '{server_name}'. Available tools: {', '.join(available_tools)}"

                        # Build detailed information
                        info = f"Tool: {server_name}.{tool_name}\n"
                        info += f"Description: {target_tool.description or 'No description available'}\n"

                        # Add parameters if available
                        if (
                            hasattr(target_tool, "inputSchema")
                            and target_tool.inputSchema
                        ):
                            info += f"\nParameters:\n"
                            if "properties" in target_tool.inputSchema:
                                for param_name, param_info in target_tool.inputSchema[
                                    "properties"
                                ].items():
                                    param_type = param_info.get("type", "unknown")
                                    param_desc = param_info.get(
                                        "description", "No description"
                                    )
                                    required = (
                                        param_name
                                        in target_tool.inputSchema.get("required", [])
                                    )
                                    required_str = (
                                        " (required)" if required else " (optional)"
                                    )
                                    info += f"  - {param_name} ({param_type}){required_str}: {param_desc}\n"
                            else:
                                info += "  No parameter details available\n"
                        else:
                            info += "\nParameters: No parameter information available\n"

                        return info
                    except Exception as e:
                        return f"Error getting tool info from '{server_name}': {e}"

                # Run the async function - use nest_asyncio to handle nested event loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    result = asyncio.run(_get_tool_info())
                    if self.debug:
                        print(result)
                    return None
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_get_tool_info())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result()
                        if self.debug:
                            print(result)
                        return None

            except Exception as e:
                return f"Error getting tool info for '{tool_identifier}': {e}"

        return get_tool_info

    @track(name="llm_completion", type="llm")
    async def _call_llm_with_span(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Call LLM with proper Opik span management."""
        try:
            if self.debug:
                print(f"🤖 Calling LLM: {model}")
                print(f"📝 Messages count: {len(messages)}")
                print(f"🔧 Tools count: {len(tools) if tools else 0}")

            # Call the LLM - Opik will automatically track this as a span within the current trace
            resp = completion(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else "none",
                **kwargs,
            )

            if self.debug:
                print(f"📊 LLM response type: {type(resp)}")
            if resp is None:
                if self.debug:
                    print("❌ LLM returned None response")
                raise ValueError("LLM returned None response")

            if not hasattr(resp, "choices"):
                if self.debug:
                    print(f"❌ LLM response missing 'choices' attribute: {resp}")
                raise ValueError(f"LLM response missing 'choices' attribute: {resp}")

            if self.debug:
                print(f"✅ LLM response has {len(resp.choices)} choices")
            return resp

        except Exception as e:
            if self.debug:
                print(f"❌ LLM call failed: {e}")
                print(f"❌ Exception type: {type(e)}")
            raise

    @track
    async def chat_once(self, user_text: str) -> str:
        if not self.sessions:
            raise RuntimeError("Not connected to any MCP servers.")

        # Update Opik context with thread_id for conversation grouping
        try:
            opik_context.update_current_trace(thread_id=self.thread_id)
        except Exception:
            # Opik not available, continue without tracing
            pass

        # 1) Fetch tool catalog from all MCP servers
        tools = await self._get_all_tools()

        # 2) Add user message to persistent history
        user_msg = _mk_user_msg(user_text)
        self.messages.append(user_msg)

        # 3) Chat loop with tool calling using persistent messages
        text_reply: str = ""

        for round_num in range(self.max_rounds):
            try:
                if self.debug:
                    print(f"🔄 LLM call round {round_num + 1}/{self.max_rounds}")

                # Show spinner while processing
                with self.console.status(
                    "[bold green]Thinking...[/bold green]", spinner="dots"
                ):
                    # Call LLM with proper span management within the current trace
                    resp = await self._call_llm_with_span(
                        model=self.model,
                        messages=self.messages,
                        tools=tools if tools else None,
                        **self.model_kwargs,
                    )

                choice = resp.choices[0].message
                # If the model just replied with text (no tool calls), return it
                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    text_reply = (choice.content or "").strip()
                    if self.debug:
                        print(
                            f"✅ LLM returned text response: {len(text_reply)} characters"
                        )
                    # Add assistant's final response to persistent history
                    self.messages.append({"role": "assistant", "content": text_reply})
                    break
                else:
                    if self.debug:
                        print(f"🔧 LLM requested {len(tool_calls)} tool calls")
            except Exception as e:
                if self.debug:
                    print(f"❌ LLM call failed in round {round_num + 1}: {e}")
                text_reply = f"Error in LLM call: {e}"
                break

            # 4) Execute each requested tool via MCP
            executed_tool_msgs: List[Dict[str, Any]] = []
            assistant_tool_stub = []

            for tc in tool_calls:
                if self.debug:
                    print(f"🔧 Executing tool: {tc.function.name}")

                # Show spinner while executing tool
                with self.console.status(
                    f"[bold blue]Executing {tc.function.name}...[/bold blue]",
                    spinner="dots",
                ):
                    content_str = await self._execute_tool_call(tc)

                if self.debug:
                    print(f"📊 Tool result length: {len(content_str)} characters")

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
                executed_tool_msgs.append(_mk_tool_result_msg(tc.id, content_str))

            # Add the assistant tool-call stub + tool results to persistent history
            if self.debug:
                print(
                    f"📝 Adding {len(executed_tool_msgs)} tool results to conversation history"
                )
            self.messages.append(_mk_assistant_tool_msg(assistant_tool_stub))
            self.messages.extend(executed_tool_msgs)
            if self.debug:
                print(f"📊 Total messages in history: {len(self.messages)}")

            # Debug: Check the last message content
            if executed_tool_msgs and self.debug:
                last_tool_result = executed_tool_msgs[-1]
                print(
                    f"🔍 Last tool result preview: {str(last_tool_result.get('content', ''))[:200]}..."
                )
                print(
                    f"🔍 Last tool result length: {len(str(last_tool_result.get('content', '')))}"
                )

        return text_reply

    def clear_messages(self):
        """Clear the message history, keeping only the system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get a copy of the current message history."""
        return self.messages.copy()

    def get_message_count(self) -> int:
        """Get the number of messages in the history (excluding system prompt)."""
        return len(self.messages) - 1  # Subtract 1 for system prompt

    async def run(self):
        """Run the complete chat session with server connections and chat loop."""
        try:
            self.console.print("[bold blue]Loaded configuration[/bold blue]")
            self.console.print(
                f"Found [bold]{len(self.servers)}[/bold] server(s) to connect to:"
            )
            for server in self.servers:
                self.console.print(
                    f"  - [cyan]{server.name}[/cyan]: {server.description}"
                )

            await self.connect_all_servers()

            if not self.sessions:
                self.console.print(
                    "[red]No servers connected successfully. Exiting.[/red]"
                )
                return

            self.console.print(
                f"\n[green]Connected to {len(self.sessions)} server(s). Ready for chat![/green]"
            )
            debug_status = "enabled" if self.debug else "disabled"
            self.console.print(f"[dim]Debug mode: {debug_status}[/dim]")
            self.console.print("[dim]Type 'quit' or 'exit' to stop.[/dim]")
            self.console.print(
                "[dim]Type '/clear' to clear conversation history.[/dim]"
            )
            self.console.print(
                "[dim]Type '/debug on' or '/debug off' to toggle debug output.[/dim]"
            )
            self.console.print(
                "[dim]Type '/show tools' to list all available tools.[/dim]"
            )
            self.console.print(
                "[dim]Type '/show tools SERVER' to list tools for a specific server.[/dim]"
            )
            self.console.print(
                "[dim]Type '/run SERVER.TOOL [args]' to execute a tool.[/dim]"
            )
            self.console.print(
                "[dim]Type '!python_code' to execute Python code (e.g., '!print(2+2)').[/dim]\n"
            )

            while True:
                try:
                    q = self.prompt_session.prompt(">>> ")
                except (EOFError, KeyboardInterrupt):
                    break

                q = q.strip()
                if q in {""}:
                    continue
                elif q.lower() in {"quit", "exit"}:
                    break
                elif q.lower() == "/clear":
                    self.clear_messages()
                    self.console.print("[yellow]Conversation history cleared.[/yellow]")
                    continue
                elif q.lower() == "/debug on":
                    self.debug = True
                    self.console.print("[green]Debug mode enabled.[/green]")
                    continue
                elif q.lower() == "/debug off":
                    self.debug = False
                    self.console.print("[yellow]Debug mode disabled.[/yellow]")
                    continue
                elif q.startswith("/show tools"):
                    # Handle /show tools and /show tools NAME commands
                    parts = q.split()
                    if len(parts) == 2:  # /show tools
                        # Show all servers and their tools
                        await self._handle_show_tools()
                    elif len(parts) == 3:  # /show tools NAME
                        # Show tools for specific server
                        server_name = parts[2]
                        await self._handle_show_tools(server_name)
                    else:
                        self.console.print(
                            "[yellow]Usage: /show tools [server_name][/yellow]"
                        )
                    continue
                elif q.startswith("/run "):
                    # Handle /run SERVER.TOOL args
                    tool_command = q[5:].strip()  # Remove "/run " prefix
                    if tool_command:
                        await self._handle_run_tool(tool_command)
                    else:
                        self.console.print(
                            "[yellow]Usage: /run SERVER.TOOL [args][/yellow]"
                        )
                    continue
                elif q.startswith("!"):
                    # Execute Python code
                    python_code = q[1:].strip()  # Remove the ! prefix
                    if python_code:
                        result = self._execute_python_code(python_code)
                        self.console.print(f"\n[bold green]Python:[/bold green]")
                        self.console.print(result)
                    else:
                        self.console.print(
                            "[yellow]No Python code provided after ![/yellow]"
                        )
                    self.console.print()  # Add spacing
                    continue

                a = await self.chat_once(q)

                # Display bot response with Rich markdown formatting
                if a:
                    self.console.print("\n[bold blue]Assistant:[/bold blue]")
                    self.console.print(Markdown(a))
                else:
                    self.console.print("[dim]Assistant: (no reply)[/dim]")
                self.console.print()  # Add spacing between exchanges
        finally:
            self.console.print("\n[dim]Shutting down ez-mcp-chatbot...[/dim]")
            await self.close()

    async def close(self):
        await self.exit_stack.aclose()

    async def _handle_show_tools(self, server_name: str = None):
        """Handle /show tools and /show tools NAME commands."""
        try:
            # If no server specified, show all servers and their tools
            if server_name is None:
                if not self.sessions:
                    self.console.print("[bold red]No MCP servers connected[/bold red]")
                    self.console.print(
                        "Use `!list_available_tools()` to see connection status."
                    )
                    return

                # Create async function to get all tools for all servers
                async def _get_all_tools_data():
                    all_data = []
                    for srv_name in self.sessions.keys():
                        try:
                            session = self.sessions[srv_name]
                            tools_resp = await session.list_tools()

                            server_data = {"name": srv_name, "tools": [], "error": None}

                            if tools_resp.tools:
                                for tool in tools_resp.tools:
                                    server_data["tools"].append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description
                                            or "No description available",
                                        }
                                    )

                            all_data.append(server_data)
                        except Exception as e:
                            all_data.append(
                                {"name": srv_name, "tools": [], "error": str(e)}
                            )

                    return all_data

                # Run the async function and then print
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                    data = asyncio.run(_get_all_tools_data())
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(_get_all_tools_data())
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        data = future.result()

                # Now print the formatted output
                self.console.print(
                    "[bold blue]# Available MCP Servers and Tools[/bold blue]"
                )
                self.console.print()

                for server_data in data:
                    if server_data["error"]:
                        self.console.print(
                            f"[bold cyan]## 📡 Server: {server_data['name']}[/bold cyan]"
                        )
                        self.console.print()
                        self.console.print(
                            f"[italic red]*Error getting tools: {server_data['error']}*[/italic red]"
                        )
                        self.console.print()
                    else:
                        self.console.print(
                            f"[bold cyan]## 📡 Server: {server_data['name']}[/bold cyan]"
                        )
                        self.console.print()

                        if server_data["tools"]:
                            for tool in server_data["tools"]:
                                self.console.print(
                                    f"[green]-[/green] [bold]{tool['name']}[/bold]: {tool['description']}"
                                )
                        else:
                            self.console.print("[italic]*No tools available*[/italic]")

                        self.console.print()

                return

            # Check if server exists
            if server_name not in self.sessions:
                available_servers = list(self.sessions.keys())
                self.console.print("[bold red]## Error[/bold red]")
                self.console.print(
                    f"Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"
                )
                return

            # Create async function to get tools for specific server
            async def _get_tools_data():
                session = self.sessions[server_name]
                try:
                    tools_resp = await session.list_tools()

                    tools_data = []
                    if tools_resp.tools:
                        for tool in tools_resp.tools:
                            tools_data.append(
                                {
                                    "name": tool.name,
                                    "description": tool.description
                                    or "No description available",
                                }
                            )

                    return tools_data
                except Exception as e:
                    raise e

            # Run the async function and then print
            try:
                import nest_asyncio

                nest_asyncio.apply()
                tools_data = asyncio.run(_get_tools_data())
            except Exception as e:
                # Fallback: try to run in a new thread with new event loop
                import concurrent.futures

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_get_tools_data())
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    tools_data = future.result()

            # Now print the formatted output
            self.console.print(
                f"[bold blue]# Tools for Server: {server_name}[/bold blue]"
            )
            self.console.print()

            if tools_data:
                for tool in tools_data:
                    self.console.print(
                        f"[green]-[/green] [bold]{tool['name']}[/bold]: {tool['description']}"
                    )
            else:
                self.console.print("[italic]*No tools available*[/italic]")

        except Exception as e:
            self.console.print("[bold red]## Error[/bold red]")
            self.console.print(f"Error getting tools: {e}")

    async def _handle_run_tool(self, tool_command: str):
        """Handle /run SERVER.TOOL [args] commands."""
        try:
            # Parse the tool command
            parts = tool_command.split()
            if not parts:
                self.console.print("[yellow]Usage: /run SERVER.TOOL [args][/yellow]")
                return

            tool_identifier = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            # Parse tool_identifier to extract server and tool names
            if "." not in tool_identifier:
                self.console.print(
                    f"[red]Error: Tool identifier must be in SERVER.TOOL format, got: {tool_identifier}[/red]"
                )
                return

            server_name, tool_name = tool_identifier.split(".", 1)

            # Check if server exists
            if server_name not in self.sessions:
                available_servers = list(self.sessions.keys())
                self.console.print(
                    f"[red]Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}[/red]"
                )
                return

            # Parse arguments into kwargs
            kwargs = {}
            for arg in args:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    kwargs[key] = value
                else:
                    # If no key=value format, treat as positional (we'll use a generic key)
                    kwargs[f"arg_{len(kwargs)}"] = arg

            # Call the tool
            session = self.sessions[server_name]
            try:
                self.console.print(
                    f"[blue]🔧 Calling tool: {tool_name} with args: {kwargs}[/blue]"
                )

                result = await session.call_tool(tool_name, kwargs)

                # Convert MCP result to readable string
                if hasattr(result, "content") and result.content is not None:
                    try:
                        content_data = result.content
                        if isinstance(content_data, list) and len(content_data) > 0:
                            if content_data[0].get("type") == "text":
                                content_str = content_data[0]["text"]
                            else:
                                content_str = str(content_data)
                        else:
                            content_str = str(content_data)
                    except Exception:
                        content_str = str(result.content)
                else:
                    content_str = str(result)

                self.console.print(
                    f"[green]✅ Tool {tool_name} completed successfully[/green]"
                )
                self.console.print(f"[green]📊 Result:[/green]")
                self.console.print(content_str)

            except Exception as e:
                self.console.print(f"[red]❌ Tool {tool_name} failed: {e}[/red]")

        except Exception as e:
            self.console.print(f"[red]Error executing tool '{tool_command}': {e}[/red]")

    def list_available_tools(self):
        """
        List all available MCP tools that can be executed.
        Returns a list of tool names and descriptions.
        """
        # Since we're in an async context, we'll provide a simpler approach
        # that works with the current sessions
        try:
            result = "Available MCP Servers and Tools:\n"

            # Get tools from each connected server
            for server_name, session in self.sessions.items():
                result += f"\n📡 Server: {server_name}\n"
                result += f"   Status: Connected\n"
                result += f"   To get tools: await self.sessions['{server_name}'].list_tools()\n"

            result += "\n🔧 Quick Tool Examples:\n"
            result += "   await self.sessions['ez-mcp-server'].call_tool('list_experiments', {'limit': 5})\n"
            result += "   await self.sessions['ez-mcp-server'].call_tool('list_projects', {})\n"
            result += "   await self.sessions['ez-mcp-server'].call_tool('get_session_info', {'random_string': 'test'})\n"

            result += "\n💡 To see all tools:\n"
            result += "   await self._get_all_tools()\n"

            return result

        except Exception as e:
            return f"Error getting tools: {e}"

    def call_session_tool(self, server_name: str, tool_name: str, **kwargs):
        """
        Directly call a tool on a specific MCP server session.
        This bypasses the tool call infrastructure for direct execution.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        import asyncio

        async def _call_tool():
            if server_name not in self.sessions:
                return f"Error: Server '{server_name}' not found. Available: {list(self.sessions.keys())}"

            session = self.sessions[server_name]
            try:
                result = await session.call_tool(tool_name, kwargs)
                return result
            except Exception as e:
                return f"Error calling tool '{tool_name}': {e}"

        try:
            loop = asyncio.get_running_loop()
            return "Use await session.call_tool() in async context"
        except RuntimeError:
            return asyncio.run(_call_tool())


def create_default_config(config_path: str = "config.json"):
    """Create a default config.json file with example configuration."""
    default_config = {
        "model": "openai/gpt-4o-mini",
        "model_kwargs": {"temperature": 0.0},
        "mcp_servers": [
            {
                "name": "ez-mcp-server",
                "description": "Ez MCP server for tool discovery and execution",
                "command": "ez-mcp-server",
                "args": [],
            }
        ],
    }

    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)

    print(f"✅ Created default configuration file: {config_path}")
    print("📝 Edit the file to customize your MCP server configuration")
    print(
        "🔧 You can add multiple servers, modify commands, and set environment variables"
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Chatbot with Opik tracing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ez-mcp-chatbot config.json        # Use specific config
  ez-mcp-chatbot --opik hosted      # Use hosted Opik instance
  ez-mcp-chatbot --opik disabled    # Disable Opik tracing
  ez-mcp-chatbot --init             # Create default config.json
  ez-mcp-chatbot --system-prompt "You are a helpful coding assistant"  # Custom system prompt
        """,
    )

    parser.add_argument(
        "config_path",
        nargs="?",
        default="config.json",
        help="Path to the configuration file (default: config.json)",
    )

    parser.add_argument(
        "--opik",
        choices=["local", "hosted", "disabled"],
        default="hosted",
        help="Opik tracing mode: local (default), hosted, or disabled",
    )

    parser.add_argument(
        "--init", action="store_true", help="Create a default config.json file and exit"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output during processing"
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the chatbot (overrides default)",
    )

    return parser.parse_args()


async def main():
    # Parse arguments
    args = parse_arguments()

    # Configure Opik based on command-line argument
    configure_opik(args.opik)

    # Use provided system prompt or default
    system_prompt = args.system_prompt if args.system_prompt else """
You are a helpful AI system for answering questions that can be answered
with any of the available tools.
"""

    bot = MCPChatbot(args.config_path, system_prompt=system_prompt, debug=args.debug)
    await bot.run()


def main_sync():
    """Synchronous entry point that handles event loop conflicts."""
    try:
        # Parse arguments first to handle --help and --init without async issues
        args = parse_arguments()

        # Handle --init flag synchronously
        if args.init:
            create_default_config(args.config_path)
            return
    except SystemExit:
        # This happens when --help is used, which is expected behavior
        return

    # Apply nest_asyncio to allow nested event loops
    import nest_asyncio

    nest_asyncio.apply()

    # Now we can safely use asyncio.run() even if there's already an event loop
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
