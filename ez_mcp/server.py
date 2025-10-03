#!/usr/bin/env python3
"""
Simple demo MCP server for testing the multi-server client.
This server provides basic math operations as tools.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from typing import Any, Dict, List

from mcp import Tool
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Import our registry and session management
from .utils import registry, load_tools_from_file
from .session import initialize_session

# SSE transport imports
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# Create the server
server = Server("ez-mcp-server")

# Global variable to track server state for clean shutdown
_server_task = None


def load_default_tools():
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ez MCP Server")
    parser.add_argument(
        "tools_file",
        nargs="?",
        default="tools.py",
        help="Path to the tools file containing functions to serve as MCP tools (default: tools.py)",
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
    return parser.parse_args()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return registry.get_tools()


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handle tool calls."""
    return registry.call_tool(name, arguments)


# SSE transport implementation
app = FastAPI(title="Ez MCP Server", version="1.0.0")

# Global variables for SSE communication
_sse_clients = set()
_message_queue = asyncio.Queue()


@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for server-to-client communication."""

    async def event_generator():
        # Add client to the set
        client_id = id(asyncio.current_task())
        _sse_clients.add(client_id)
        print(f"🔌 SSE client connected: {client_id}")

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
            print(f"🔌 SSE client disconnected: {client_id}")
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
async def message_endpoint(request: Request):
    """HTTP POST endpoint for client-to-server communication."""
    try:
        data = await request.json()
        print(f"📨 Received message: {data}")

        # Process the message through the MCP server
        # This is a simplified implementation - in a real scenario,
        # you'd need to properly handle MCP protocol messages
        response = {"status": "received", "data": data}

        # Send response back via SSE
        await _message_queue.put(response)

        return {"status": "success", "message": "Message processed"}
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "transport": "sse"}


async def start_sse_server(host: str, port: int):
    """Start the SSE server."""
    print(f"🚀 Starting SSE server on {host}:{port}")
    print(f"📡 SSE endpoint: http://{host}:{port}/sse")
    print(f"📨 Messages endpoint: http://{host}:{port}/messages")
    print(f"🏥 Health check: http://{host}:{port}/health")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\n🛑 Received shutdown signal, cleaning up...")
    if _server_task and not _server_task.done():
        _server_task.cancel()
    # Force immediate exit to avoid waiting for stdin
    os._exit(0)


async def main():
    """Run the server."""
    global _server_task

    # Parse command line arguments
    args = parse_args()

    print("🚀 Ez MCP Server Starting...")
    print(f"🚌 Transport: {args.transport}")
    print(f"📁 Tools file: {args.tools_file}")

    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load tools from the specified file or use default tools
    try:
        if args.tools_file == "tools.py" and not os.path.exists(args.tools_file):
            # No tools file provided and default doesn't exist, load example tools
            load_default_tools()
            print("✓ Loaded default example tools from README")
        else:
            load_tools_from_file(args.tools_file)
            print(f"✓ Loaded tools from {args.tools_file}")
    except Exception as e:
        print(f"❌ Failed to load tools from {args.tools_file}: {e}")
        sys.exit(1)

    # Initialize session context
    try:
        initialize_session()
        print("✓ Session initialized")
    except Exception as e:
        print(f"⚠️  Session initialization failed: {e}")

    print(f"🔧 Available tools: {[tool.name for tool in registry.get_tools()]}")

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
            await start_sse_server(args.host, args.port)
        else:
            print(f"❌ Unknown transport: {args.transport}")
            sys.exit(1)
    except asyncio.CancelledError:
        print("🛑 Server shutdown completed")
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt, shutting down...")
        # Force immediate exit to avoid waiting for stdin
        os._exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


def main_sync():
    """Synchronous entry point for the ez-mcp command."""
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
