import asyncio
import sys
import json
import os
import uuid
import argparse
import io
import traceback
import warnings
import ast
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from opik import track
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from .utils import (
    configure_opik as configure_opik_util,
    call_llm_with_tracing,
    resolve_prompt_with_opik,
    chat_with_tools,
)
from .mcp_utils import MCPManager

# Suppress litellm RuntimeWarning about coroutines never awaited
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="coroutine 'close_litellm_async_clients' was never awaited",
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*"
)

load_dotenv()


def configure_opik(opik_mode: str = "hosted") -> None:
    """Configure Opik based on the specified mode."""
    configure_opik_util(opik_mode, "ez-mcp-chatbot")


class ChatbotCompleter(Completer):
    """Custom completer for the chatbot with command and Python code completion."""

    def __init__(self) -> None:
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
            "self.mcp_manager.sessions",
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

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        """Provide completions based on the current input."""
        text = document.text
        # word = document.get_word_before_cursor()  # Unused variable

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
        max_rounds: Optional[int] = 10,
        debug: bool = False,
        model_override: Optional[str] = None,
        model_args_override: Optional[Dict[str, Any]] = None,
        tools_file: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.config_path = config_path
        self.tools_file = tools_file
        self.prompt_id = prompt_id
        self.servers, self.model, self.model_parameters = self.load_config(config_path)

        # Override model if provided
        if model_override:
            self.model = model_override

        # Override model parameters if provided
        if model_args_override:
            self.model_parameters = model_args_override
        self.max_rounds = max_rounds
        self.debug = debug
        self.console = Console()
        # Generate unique thread-id for this chatbot instance
        self.thread_id = str(uuid.uuid4())
        self.clear_messages()

        # Initialize MCP manager
        self.mcp_manager = MCPManager(console=self.console, debug=debug)

        # Set up prompt_toolkit for enhanced input handling
        self._setup_prompt_toolkit()

        # Set up persistent Python evaluation environment
        self._setup_python_environment()

    def _setup_prompt_toolkit(self) -> None:
        """Set up prompt_toolkit for enhanced input handling with history and completion."""
        # Set up history file
        history_file = os.path.expanduser("~/.opik_mcp_chatbot_history")

        # Create prompt session with history and completion
        self.prompt_session: PromptSession = PromptSession(
            history=FileHistory(history_file),
            completer=ChatbotCompleter(),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True,
        )

    def _setup_python_environment(self) -> None:
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
        config_path: str = "ez-config.json",
    ) -> tuple[List, str, Dict[str, Any]]:
        """Load configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Use default configuration when no config file exists
            config = {
                "model": "openai/gpt-4o-mini",
                "model_parameters": {"temperature": 0.2},
                "mcp_servers": [
                    {
                        "name": "ez-mcp-server",
                        "description": "Ez MCP server with default tools",
                        "command": "ez-mcp-server",
                        "args": [],
                    }
                ],
            }

        # Extract model configuration (support both model_parameters and model_kwargs for backwards compatibility)
        model = config.get("model", "openai/gpt-4o-mini")
        # Prefer model_parameters, fall back to model_kwargs for backwards compatibility
        if "model_parameters" in config:
            model_parameters = config["model_parameters"]
        elif "model_kwargs" in config:
            warnings.warn(
                "Config file uses 'model_kwargs' which is deprecated. "
                "Please update to use 'model_parameters' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_parameters = config["model_kwargs"]
        else:
            model_parameters = {"temperature": 0.2}

        return config.get("mcp_servers", []), model, model_parameters

    async def connect_all_servers(self) -> None:
        """Connect to all configured MCP servers via subprocess."""
        # Load MCP configuration and connect
        if self.tools_file:
            # Resolve tools file path (download from URL if necessary)
            from .utils import resolve_tools_file_path

            resolved_tools_file = resolve_tools_file_path(self.tools_file, self.console)

            # Create dynamic MCP server configuration using tools file
            from .mcp_utils import ServerConfig

            servers = [
                ServerConfig(
                    name="ez-mcp-server",
                    description="Ez MCP server for tool discovery and execution",
                    command="ez-mcp-server",
                    args=[resolved_tools_file],
                )
            ]
            self.console.print(
                f"📡 Created MCP server configuration with tools file: {resolved_tools_file}"
            )
            await self.mcp_manager.connect_all_servers(servers)
        else:
            servers = self.mcp_manager.load_mcp_config(self.config_path)
            if servers:
                await self.mcp_manager.connect_all_servers(servers)

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
                        except Exception:
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

        except Exception:
            # Get the full traceback for better error reporting
            error_msg = traceback.format_exc()
            return f"🐍 Python Error:\n{error_msg}"

    def _create_await_helper(self) -> Any:
        """Create a helper function to run async code from sync context."""

        def await_helper(coro: Any) -> Any:
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

    def _create_direct_tool_runner(self) -> Any:
        """Create a helper function to run tools via MCP sessions."""

        def run_tool(tool_identifier: str, **kwargs: Any) -> str:
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
                if server_name not in self.mcp_manager.sessions:
                    available_servers = list(self.mcp_manager.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to call the tool
                async def _call_tool() -> Any:
                    session = self.mcp_manager.sessions[server_name]
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
                                    # Type guard for MCP content objects
                                    if (
                                        hasattr(content_data[0], "type")
                                        and content_data[0].type == "text"
                                    ):
                                        return content_data[0].text
                                    elif (
                                        isinstance(content_data[0], dict)
                                        and content_data[0].get("type") == "text"
                                    ):
                                        return content_data[0]["text"]
                                    else:
                                        # Extract text content from list of content items
                                        text_parts = []
                                        for item in content_data:
                                            try:
                                                if hasattr(item, "text"):
                                                    text_parts.append(item.text)
                                                elif (
                                                    isinstance(item, dict)
                                                    and "text" in item
                                                ):
                                                    text_parts.append(item["text"])
                                                else:
                                                    text_parts.append(str(item))
                                            except Exception:
                                                text_parts.append(str(item))
                                        return "".join(text_parts)
                                else:
                                    return str(content_data)
                            except Exception:
                                return str(result.content)
                        else:
                            return str(result)
                    except Exception as e:
                        return f"Error calling tool '{tool_name}': {e}"

                # Run the async function using common utility
                try:
                    from .utils import run_async_in_sync_context

                    result = run_async_in_sync_context(_call_tool)
                    if self.debug:
                        print(result)
                    return (
                        str(result)
                        if result is not None
                        else "Tool execution completed"
                    )
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread() -> Any:
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
                        return (
                            str(result)
                            if result is not None
                            else "Tool execution completed"
                        )

            except Exception as e:
                return f"Error executing tool '{tool_identifier}': {e}"

        return run_tool

    def _create_direct_tool_runner_return(self) -> Any:
        """Create a helper function to run tools via MCP sessions that returns the result."""

        def run_tool_return(tool_identifier: str, **kwargs: Any) -> Any:
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
                if server_name not in self.mcp_manager.sessions:
                    available_servers = list(self.mcp_manager.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to call the tool
                async def _call_tool() -> Any:
                    session = self.mcp_manager.sessions[server_name]
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
                                    # Type guard for MCP content objects
                                    if (
                                        hasattr(content_data[0], "type")
                                        and content_data[0].type == "text"
                                    ):
                                        return content_data[0].text
                                    elif (
                                        isinstance(content_data[0], dict)
                                        and content_data[0].get("type") == "text"
                                    ):
                                        return content_data[0]["text"]
                                    else:
                                        # Extract text content from list of content items
                                        text_parts = []
                                        for item in content_data:
                                            try:
                                                if hasattr(item, "text"):
                                                    text_parts.append(item.text)
                                                elif (
                                                    isinstance(item, dict)
                                                    and "text" in item
                                                ):
                                                    text_parts.append(item["text"])
                                                else:
                                                    text_parts.append(str(item))
                                            except Exception:
                                                text_parts.append(str(item))
                                        return "".join(text_parts)
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

                    def run_in_thread() -> Any:
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

    def _create_direct_tool_getter(self) -> Any:
        """Create a helper function to get tools via MCP sessions."""

        def get_tools(server_name: Optional[str] = None) -> str:
            """Helper to get tools - prints rich formatted list of tools."""
            try:
                # If no server specified, show all servers and their tools
                if server_name is None:
                    if not self.mcp_manager.sessions:
                        self.console.print(
                            "[bold red]No MCP servers connected[/bold red]"
                        )
                        self.console.print(
                            "Use `!list_available_tools()` to see connection status."
                        )
                        return "No MCP servers connected"

                    # Create async function to get all tools for all servers
                    async def _get_all_tools_data() -> Any:
                        all_data = []
                        for srv_name in self.mcp_manager.sessions.keys():
                            try:
                                session = self.mcp_manager.sessions[srv_name]
                                tools_resp = await session.list_tools()

                                server_data: Dict[str, Any] = {
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

                        def run_in_thread() -> Any:
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

                    return "Tools listed successfully"

                # Check if server exists
                if server_name not in self.mcp_manager.sessions:
                    available_servers = list(self.mcp_manager.sessions.keys())
                    self.console.print("[bold red]## Error[/bold red]")
                    self.console.print(
                        f"Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"
                    )
                    return "Tools listed successfully"

                # Create async function to get tools for specific server
                async def _get_tools_data() -> Any:
                    session = self.mcp_manager.sessions[server_name]
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
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread() -> Any:
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

                return "Tools listed successfully"

            except Exception as e:
                self.console.print("[bold red]## Error[/bold red]")
                self.console.print(f"Error getting tools: {e}")
                return "Tools listed successfully"

        return get_tools

    def _create_direct_tool_info_getter(self) -> Any:
        """Create a helper function to get detailed tool information via MCP sessions."""

        def get_tool_info(tool_identifier: str) -> str:
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
                if server_name not in self.mcp_manager.sessions:
                    available_servers = list(self.mcp_manager.sessions.keys())
                    return f"Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"

                # Create async function to get tool info
                async def _get_tool_info() -> Any:
                    session = self.mcp_manager.sessions[server_name]
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
                            info += "\nParameters:\n"
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
                    return "Tools listed successfully"
                except Exception:
                    # Fallback: try to run in a new thread with new event loop
                    import concurrent.futures

                    def run_in_thread() -> Any:
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
                        return (
                            str(result)
                            if result is not None
                            else "Tool execution completed"
                        )

            except Exception as e:
                return f"Error getting tool info for '{tool_identifier}': {e}"

        return get_tool_info

    @track(name="llm_completion", type="llm")
    async def _call_llm_with_span(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call LLM with proper Opik span management."""
        # Use the common utility function for consistent LLM processing
        return call_llm_with_tracing(
            model=model,
            messages=messages,
            tools=tools,
            debug=self.debug,
            console=None,  # Use print for chatbot to maintain consistency
            **kwargs,
        )

    @track
    async def chat(self, user_text: str) -> str:
        """Chat method that delegates to the shared chat_with_tools function."""
        return await chat_with_tools(
            user_text=user_text,
            system_prompt=self.system_prompt,
            model=self.model,
            model_parameters=self.model_parameters,
            mcp_manager=self.mcp_manager,
            messages=self.messages,
            max_rounds=self.max_rounds or 4,
            debug=self.debug,
            console=self.console,
            thread_id=self.thread_id,
            prompt_id=self.prompt_id,
        )

    def clear_messages(self) -> None:
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

    async def run(self) -> None:
        """Run the complete chat session with server connections and chat loop."""
        try:
            self.console.print("[bold blue]Loaded configuration[/bold blue]")
            self.console.print(
                f"Found [bold]{len(self.servers)}[/bold] server(s) to connect to:"
            )
            for server in self.servers:
                self.console.print(
                    f"  - [cyan]{server['name']}[/cyan]: {server['description']}"
                )

            await self.connect_all_servers()

            if not self.mcp_manager.sessions:
                self.console.print(
                    "[red]No servers connected successfully. Exiting.[/red]"
                )
                return

            self.console.print(
                f"\n[green]Connected to {len(self.mcp_manager.sessions)} server(s). Ready for chat![/green]"
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
                "[dim]Type '/run SERVER.TOOL {'key': 'value'}' to execute a tool with JSON/dict arguments.[/dim]"
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
                            "[yellow]Usage: /run SERVER.TOOL {'key': 'value'} or /run SERVER.TOOL {\"key\": \"value\"}[/yellow]"
                        )
                    continue
                elif q.startswith("!"):
                    # Execute Python code
                    python_code = q[1:].strip()  # Remove the ! prefix
                    if python_code:
                        result = self._execute_python_code(python_code)
                        self.console.print("\n[bold green]Python:[/bold green]")
                        self.console.print(result)
                    else:
                        self.console.print(
                            "[yellow]No Python code provided after ![/yellow]"
                        )
                    self.console.print()  # Add spacing
                    continue

                a: str = await self.chat(str(q))

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

    async def close(self) -> None:
        await self.mcp_manager.close()

    async def _handle_show_tools(self, server_name: Optional[str] = None) -> None:
        """Handle /show tools and /show tools NAME commands."""
        try:
            # If no server specified, show all servers and their tools
            if server_name is None:
                if not self.mcp_manager.sessions:
                    self.console.print("[bold red]No MCP servers connected[/bold red]")
                    self.console.print(
                        "Use `!list_available_tools()` to see connection status."
                    )
                    return

                # Create async function to get all tools for all servers
                async def _get_all_tools_data() -> Any:
                    all_data = []
                    for srv_name in self.mcp_manager.sessions.keys():
                        try:
                            session = self.mcp_manager.sessions[srv_name]
                            tools_resp = await session.list_tools()

                            server_data: Dict[str, Any] = {
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

                    def run_in_thread() -> Any:
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
            if server_name not in self.mcp_manager.sessions:
                available_servers = list(self.mcp_manager.sessions.keys())
                self.console.print("[bold red]## Error[/bold red]")
                self.console.print(
                    f"Server '{server_name}' not found. Available servers: {', '.join(available_servers)}"
                )
                return

            # Create async function to get tools for specific server
            async def _get_tools_data() -> Any:
                session = self.mcp_manager.sessions[server_name]
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
            except Exception:
                # Fallback: try to run in a new thread with new event loop
                import concurrent.futures

                def run_in_thread() -> Any:
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

    async def _handle_run_tool(self, tool_command: str) -> None:
        """Handle /run SERVER.TOOL [args] commands.

        Args format: JSON dict or Python dict literal, e.g.:
        /run SERVER.TOOL {"key": "value", "num": 1}
        /run SERVER.TOOL {'key': 'value', 'num': 1}
        """
        try:
            # Parse the tool command
            parts = tool_command.split(maxsplit=1)
            if not parts:
                self.console.print(
                    "[yellow]Usage: /run SERVER.TOOL {'key': 'value'} or /run SERVER.TOOL {\"key\": \"value\"}[/yellow]"
                )
                return

            tool_identifier = parts[0]
            args_str = parts[1] if len(parts) > 1 else "{}"

            # Parse tool_identifier to extract server and tool names
            if "." not in tool_identifier:
                self.console.print(
                    f"[red]Error: Tool identifier must be in SERVER.TOOL format, got: {tool_identifier}[/red]"
                )
                return

            server_name, tool_name = tool_identifier.split(".", 1)

            # Check if server exists
            if server_name not in self.mcp_manager.sessions:
                available_servers = list(self.mcp_manager.sessions.keys())
                self.console.print(
                    f"[red]Error: Server '{server_name}' not found. Available servers: {', '.join(available_servers)}[/red]"
                )
                return

            # Parse arguments as JSON or Python dict literal
            kwargs = {}
            if args_str.strip():
                try:
                    # First try to parse as JSON (double quotes)
                    kwargs = json.loads(args_str)
                except json.JSONDecodeError:
                    try:
                        # If JSON fails, try to parse as Python dict literal (single quotes)
                        # Use ast.literal_eval for safe evaluation
                        parsed = ast.literal_eval(args_str)
                        if isinstance(parsed, dict):
                            kwargs = parsed
                        else:
                            self.console.print(
                                f"[yellow]Warning: Arguments should be a dict, got {type(parsed).__name__}. Using empty dict.[/yellow]"
                            )
                            kwargs = {}
                    except (ValueError, SyntaxError) as e:
                        self.console.print(
                            f"[red]Error: Invalid arguments format. Expected JSON dict or Python dict literal. Error: {e}[/red]"
                        )
                        self.console.print(
                            "[yellow]Example: /run SERVER.TOOL {'key': 'value', 'num': 1}[/yellow]"
                        )
                        return

            # Call the tool
            session = self.mcp_manager.sessions[server_name]
            try:
                self.console.print(
                    f"[blue]🔧 Calling tool: {tool_name} with args: {kwargs}[/blue]"
                )

                result = await session.call_tool(tool_name, kwargs)

                # Process result using shared utility function
                from .utils import process_mcp_tool_result

                processed_result = process_mcp_tool_result(
                    result, tool_name, self.debug
                )

                self.console.print(
                    f"[green]✅ Tool {tool_name} completed successfully[/green]"
                )
                self.console.print("[green]📊 Result:[/green]")

                # Display the result appropriately based on its type
                if isinstance(processed_result, (dict, list)):
                    # For structured data, pretty print as JSON
                    self.console.print(json.dumps(processed_result, indent=2))
                else:
                    # For strings and other types, display as-is
                    self.console.print(str(processed_result))

            except Exception as e:
                self.console.print(f"[red]❌ Tool {tool_name} failed: {e}[/red]")

        except Exception as e:
            self.console.print(f"[red]Error executing tool '{tool_command}': {e}[/red]")

    def list_available_tools(self) -> str:
        """
        List all available MCP tools that can be executed.
        Returns a list of tool names and descriptions.
        """
        # Since we're in an async context, we'll provide a simpler approach
        # that works with the current sessions
        try:
            result = "Available MCP Servers and Tools:\n"

            # Get tools from each connected server
            for server_name, session in self.mcp_manager.sessions.items():
                result += f"\n📡 Server: {server_name}\n"
                result += "   Status: Connected\n"
                result += f"   To get tools: await self.mcp_manager.sessions['{server_name}'].list_tools()\n"

            result += "\n🔧 Quick Tool Examples:\n"
            result += "   await self.mcp_manager.sessions['ez-mcp-server'].call_tool('list_experiments', {'limit': 5})\n"
            result += "   await self.mcp_manager.sessions['ez-mcp-server'].call_tool('list_projects', {})\n"
            result += "   await self.mcp_manager.sessions['ez-mcp-server'].call_tool('get_session_info', {'random_string': 'test'})\n"

            result += "\n💡 To see all tools:\n"
            result += "   await self._get_all_tools()\n"

            return result

        except Exception as e:
            return f"Error getting tools: {e}"

    def call_session_tool(self, server_name: str, tool_name: str, **kwargs: Any) -> str:
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

        async def _call_tool() -> Any:
            if server_name not in self.mcp_manager.sessions:
                return f"Error: Server '{server_name}' not found. Available: {list(self.mcp_manager.sessions.keys())}"

            session = self.mcp_manager.sessions[server_name]
            try:
                result = await session.call_tool(tool_name, kwargs)
                return result
            except Exception as e:
                return f"Error calling tool '{tool_name}': {e}"

        try:
            # loop = asyncio.get_running_loop()  # Unused variable
            asyncio.get_running_loop()
            return "Use await session.call_tool() in async context"
        except RuntimeError:
            return asyncio.run(_call_tool())


def create_default_config(config_path: str = "ez-config.json") -> None:
    """Create a default ez-config.json file with example configuration."""
    default_config = {
        "model": "openai/gpt-4o-mini",
        "model_parameters": {"temperature": 0.0},
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


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Chatbot with Opik tracing support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ez-mcp-chatbot ez-config.json     # Use specific config
  ez-mcp-chatbot --opik hosted      # Use hosted Opik instance
  ez-mcp-chatbot --opik disabled    # Disable Opik tracing
  ez-mcp-chatbot --init             # Create default ez-config.json
  ez-mcp-chatbot --prompt "You are a helpful coding assistant"  # Direct string
  ez-mcp-chatbot --prompt ./my_prompt.txt                     # Load from file
  ez-mcp-chatbot --prompt my_optimized_prompt                  # Load from Opik
  ez-mcp-chatbot --model "openai/gpt-4"  # Override model from config
  ez-mcp-chatbot --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'  # Override model parameters
  ez-mcp-chatbot --tools-file "my_tools.py"  # Use custom tools file
        """,
    )

    parser.add_argument(
        "config_path",
        nargs="?",
        default="ez-config.json",
        help="Path to the configuration file (default: ez-config.json)",
    )

    parser.add_argument(
        "--opik",
        choices=["local", "hosted", "disabled"],
        default="hosted",
        help="Opik tracing mode: local (default), hosted, or disabled",
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Create a default ez-config.json file and exit",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output during processing"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom system prompt for the chatbot (overrides default). Can be a direct string, file path, or Opik prompt name.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Override the model specified in the config file",
    )

    parser.add_argument(
        "--tools-file",
        type=str,
        help="Path to a Python file containing tool definitions, or URL to download the file from. If provided, will create an MCP server configuration using this file.",
    )

    parser.add_argument(
        "--model-parameters",
        type=str,
        help='JSON string of additional keyword arguments to pass to the LLM model (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
    )

    parser.add_argument(
        "--model-kwargs",
        type=str,
        help=argparse.SUPPRESS,  # Hide from help, but keep for backwards compatibility
    )

    return parser.parse_args()


async def main() -> None:
    # Parse arguments
    args = parse_arguments()

    # Configure Opik based on command-line argument
    configure_opik(args.opik)

    # Parse model parameters JSON (with backwards compatibility for model-kwargs)
    model_args_override = None
    if args.model_parameters and args.model_kwargs:
        warnings.warn(
            "Both --model-parameters and --model-kwargs were provided. "
            "Using --model-parameters and ignoring --model-kwargs.",
            UserWarning,
            stacklevel=2,
        )
    if args.model_parameters:
        try:
            model_args_override = json.loads(args.model_parameters)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --model-parameters: {e}")
            sys.exit(1)
    elif args.model_kwargs:
        # Backwards compatibility: issue deprecation warning
        warnings.warn(
            "--model-kwargs is deprecated and will be removed in a future version. "
            "Please use --model-parameters instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            model_args_override = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --model-kwargs: {e}")
            sys.exit(1)

    # Resolve system prompt using Opik if available, otherwise use direct value
    console = Console()
    system_prompt: str
    prompt_id: Optional[str] = None

    if args.prompt:
        try:
            # Try to get Opik client for prompt resolution
            from opik import Opik

            client = Opik()
            resolved_prompt, prompt_id = resolve_prompt_with_opik(
                client, args.prompt, console
            )
            system_prompt = (
                resolved_prompt if resolved_prompt is not None else args.prompt
            )
            if prompt_id:
                console.print(f"📋 Prompt ID: {prompt_id}")
        except Exception as e:
            # If Opik is not available or fails, use the prompt value directly
            console.print(f"⚠️  Opik not available ({e}), using prompt as direct string")
            system_prompt = args.prompt
            prompt_id = None
    else:
        # Use default system prompt
        system_prompt = """
You are a helpful AI system for answering questions that can be answered
with any of the available tools.
"""

    bot = MCPChatbot(
        args.config_path,
        system_prompt=system_prompt,
        debug=args.debug,
        model_override=args.model,
        model_args_override=model_args_override,
        tools_file=args.tools_file,
        prompt_id=prompt_id,
    )
    await bot.run()


def main_sync() -> None:
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
