#!/usr/bin/env python3
"""
Shared MCP utilities for both chatbot and evaluator.
"""

import json
import os
import subprocess
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console
from opik import track


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    name: str
    description: str
    command: str
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


class MCPManager:
    """Shared MCP server management functionality."""

    def __init__(self, console: Optional[Console] = None, debug: bool = False):
        self.console = console or Console()
        self.debug = debug
        self.sessions: Dict[str, ClientSession] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.exit_stack = AsyncExitStack()
        self.thread_id = str(uuid.uuid4())

    def load_mcp_config(
        self, config_path: str = "ez-config.json"
    ) -> List[ServerConfig]:
        """Load MCP server configuration from JSON file."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Use default configuration when no config file exists
            config = {
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

        return servers

    async def connect_all_servers(self, servers: List[ServerConfig]):
        """Connect to all configured MCP servers via subprocess."""
        if not servers:
            return

        for server_config in servers:
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
                    tool["function"]["name"] = (
                        f"{server_name}_{tool['function']['name']}"
                    )
                all_tools.extend(server_tools)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning:[/yellow] Failed to get tools from [bold]{server_name}[/bold]: {e}"
                )
        return all_tools

    @track(name="execute_tool_call", type="tool")
    async def execute_tool_call(self, tool_call) -> str:
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
                                    return f"Image result from {actual_tool_name} (base64 data)"
                            except (json.JSONDecodeError, AttributeError):
                                pass
                        # If not an image, convert to string
                        if self.debug:
                            print("📝 Converting content to string")
                        content_str = str(content_data)
                    elif (
                        isinstance(content_data, dict)
                        and content_data.get("type") == "image_result"
                        and "image_base64" in content_data
                    ):
                        # Handle image result specially
                        if self.debug:
                            print(f"🖼️  Detected image result from {actual_tool_name}")
                        return f"Image result from {actual_tool_name} (base64 data)"
                    else:
                        if self.debug:
                            print("📝 Converting content to JSON string")
                        content_str = json.dumps(content_data, separators=(",", ":"))
                except Exception as e:
                    if self.debug:
                        print(f"❌ Error processing content: {e}")
                    content_str = str(result.content)
            else:
                if self.debug:
                    print("📝 Converting result to string")
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

            return content_str
        except Exception as e:
            if self.debug:
                print(f"❌ Tool {actual_tool_name} failed: {e}")
            return f"Error executing tool '{actual_tool_name}': {e}"

    async def close(self):
        """Close all MCP server connections."""
        await self.exit_stack.aclose()


def _mcp_tools_to_openai_tools(tools_resp) -> List[Dict[str, Any]]:
    """Map MCP tool spec to OpenAI function tools."""
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
