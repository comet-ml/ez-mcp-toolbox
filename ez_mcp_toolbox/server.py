import argparse
import asyncio
import json
import os
import re
import signal
import sys
from typing import Any, Dict, List, Optional, Set

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Import our registry and session management
from .utils import registry, load_tools_from_file, load_tools_from_module
from .session import initialize_session

# SSE transport imports
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# Create the server
server = Server("ez-mcp-server")

# Global variable to track server state for clean shutdown
_server_task: Optional[asyncio.Task[None]] = None

# Global variable to track quiet mode for SSE endpoints
_quiet_mode: bool = False


def load_default_tools() -> None:
    """Load the default example tools from the README."""
    from .utils import registry

    def add_numbers(a: float, b: float) -> float:
        """
        Add two numbers together.

        Args:
            a: First number to add
            b: Second number to add

        Returns:
            The sum of a and b
        """
        return a + b

    def greet_user(name: str) -> str:
        """
        Greet a user with a welcoming message.

        Args:
            name: The name of the person to greet

        Returns:
            A personalized greeting message
        """
        return f"Welcome to ez-mcp-server, {name}!"

    # Register the default tools
    registry.tool(add_numbers)
    registry.tool(greet_user)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ez MCP Server")
    parser.add_argument(
        "tools_file",
        nargs="?",
        default="DEMO",
        help="Path to tools file, module name, URL to download from, or 'none' to disable tools (e.g., 'my_tools.py', 'opik_optimizer.utils.core', 'https://example.com/tools.py', or 'none') (default: DEMO)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport method to use (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--include",
        type=str,
        help="Python regex pattern to include only matching tool names",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Python regex pattern to exclude matching tool names",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output messages",
    )
    return parser.parse_args()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return registry.get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls with timeout to prevent hangs."""
    import asyncio

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(registry.call_tool, name, arguments), timeout=25.0
        )
    except asyncio.TimeoutError:
        return [{"type": "text", "text": f"Error calling tool {name}: timeout"}]


# SSE transport implementation
app = FastAPI(title="Ez MCP Server", version="1.0.0")

# Global variables for SSE communication
_sse_clients: Set[int] = set()
_message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()


@app.get("/sse")
async def sse_endpoint() -> StreamingResponse:
    """SSE endpoint for server-to-client communication."""

    async def event_generator() -> Any:
        # Add client to the set
        client_id = id(asyncio.current_task())
        _sse_clients.add(client_id)
        if not _quiet_mode:
            print(f"🔌 SSE client connected: {client_id}", file=sys.stderr)

        try:
            while True:
                # Wait for messages from the message queue
                try:
                    message = await asyncio.wait_for(_message_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        except asyncio.CancelledError:
            if not _quiet_mode:
                print(f"🔌 SSE client disconnected: {client_id}", file=sys.stderr)
        finally:
            _sse_clients.discard(client_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@app.post("/messages")
async def message_endpoint(request: Request) -> Dict[str, Any]:
    """HTTP POST endpoint for client-to-server communication."""
    try:
        data = await request.json()
        if not _quiet_mode:
            print(f"📨 Received message: {data}", file=sys.stderr)

        # Process the message through the MCP server
        # This is a simplified implementation - in a real scenario,
        # you'd need to properly handle MCP protocol messages
        response = {"status": "received", "data": data}

        # Send response back via SSE
        await _message_queue.put(response)

        return {"status": "success", "message": "Message processed"}
    except Exception as e:
        if not _quiet_mode:
            print(f"❌ Error processing message: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "transport": "sse"}


async def start_sse_server(host: str, port: int, quiet_mode: bool = False) -> None:
    """Start the SSE server."""
    if not quiet_mode:
        print(f"🚀 Starting SSE server on {host}:{port}", file=sys.stderr)
        print(f"📡 SSE endpoint: http://{host}:{port}/sse", file=sys.stderr)
        print(f"📨 Messages endpoint: http://{host}:{port}/messages", file=sys.stderr)
        print(f"🏥 Health check: http://{host}:{port}/health", file=sys.stderr)

    log_level = "error" if quiet_mode else "info"
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    # Check for quiet mode before printing
    if not _quiet_mode:
        print("\n🛑 Received shutdown signal, cleaning up...", file=sys.stderr)
    if _server_task and not _server_task.done():
        _server_task.cancel()
    # Force immediate exit to avoid waiting for stdin
    os._exit(0)  # This never returns


async def main() -> None:
    """Run the server."""
    global _server_task, _quiet_mode

    # Parse command line arguments
    args = parse_args()

    # Determine quiet mode (from command line flag or environment variable)
    quiet_mode = bool(args.quiet) or os.getenv("EZ_MCP_QUIET") == "1"
    _quiet_mode = quiet_mode

    # Set environment variable immediately so utility functions can check it
    if quiet_mode:
        os.environ["EZ_MCP_QUIET"] = "1"

    if not quiet_mode:
        print("🚀 Ez MCP Server Starting...", file=sys.stderr)
        print(f"🚌 Transport: {args.transport}", file=sys.stderr)
        print(f"📁 Tools file: {args.tools_file}", file=sys.stderr)

    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load tools from the specified source or use default tools
    try:
        if args.tools_file == "none":
            # Skip loading any tools
            if not quiet_mode:
                print("✓ No tools loaded (tools_file set to 'none')", file=sys.stderr)
        elif args.tools_file == "DEMO" and not os.path.exists(args.tools_file):
            # No tools file provided and default doesn't exist, load example tools
            load_default_tools()
            if not quiet_mode:
                print("✓ Loaded default example tools from README", file=sys.stderr)
        elif args.tools_file.endswith(".py"):
            # Treat as file path
            load_tools_from_file(args.tools_file)
            if not quiet_mode:
                print(f"✓ Loaded tools from {args.tools_file}", file=sys.stderr)
        else:
            # Could be URL or module name - use common utility to resolve
            from .utils import resolve_tools_file_path

            resolved_tools_file = resolve_tools_file_path(args.tools_file)

            # Check if it was a URL (resolved path will be different from original)
            if resolved_tools_file != args.tools_file:
                # It was a URL that got downloaded
                load_tools_from_file(resolved_tools_file)
                if not quiet_mode:
                    print(
                        f"✓ Loaded tools from URL: {args.tools_file}", file=sys.stderr
                    )
            else:
                # It's a module name
                load_tools_from_module(args.tools_file)
                if not quiet_mode:
                    print(
                        f"✓ Loaded tools from module {args.tools_file}", file=sys.stderr
                    )
    except Exception as e:
        if not quiet_mode:
            print(
                f"❌ Failed to load tools from {args.tools_file}: {e}", file=sys.stderr
            )
        sys.exit(1)

    # Apply tool filtering if specified
    if args.include or args.exclude:
        try:
            registry.filter_tools(args.include, args.exclude)
            if not quiet_mode:
                print(
                    f"✓ Applied tool filtering (include: {args.include}, exclude: {args.exclude})",
                    file=sys.stderr,
                )
        except re.error as e:
            if not quiet_mode:
                print(f"❌ Invalid regex pattern: {e}", file=sys.stderr)
            sys.exit(1)

    # Initialize session context
    try:
        initialize_session()
        if not quiet_mode:
            print("✓ Session initialized", file=sys.stderr)
    except Exception as e:
        if not quiet_mode:
            print(f"⚠️  Session initialization failed: {e}", file=sys.stderr)

    if not quiet_mode:
        print(
            f"🔧 Available tools: {[tool.name for tool in registry.get_tools()]}",
            file=sys.stderr,
        )

    try:
        if args.transport == "stdio":
            # Use stdio transport (default)
            async with stdio_server() as (read_stream, write_stream):
                _server_task = asyncio.create_task(
                    server.run(
                        read_stream,
                        write_stream,
                        server.create_initialization_options(),
                    )
                )
                await _server_task
        elif args.transport == "sse":
            # Use SSE transport
            await start_sse_server(args.host, args.port, quiet_mode)
        else:
            if not quiet_mode:
                print(f"❌ Unknown transport: {args.transport}")
            sys.exit(1)
    except asyncio.CancelledError:
        if not quiet_mode:
            print("🛑 Server shutdown completed", file=sys.stderr)
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except KeyboardInterrupt:
        if not quiet_mode:
            print("\n🛑 Received keyboard interrupt, shutting down...", file=sys.stderr)
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except Exception as e:
        if not quiet_mode:
            print(f"❌ Server error: {e}", file=sys.stderr)
        raise


def main_sync() -> None:
    """Synchronous entry point for the ez-mcp command."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
