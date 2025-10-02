# Instant MCP

This package contains two command-line utilities:

1. `instant-mcp-server` - turn a file of Python functions into a MCP server
2. `instant-mcp-chatbot` - interactively debug MCP servers, with traces logged to Opik

## Installation

```
pip install instant-mcp --upgrade
```

## Quick start

```
instant-mcp-chatbot
```

That will start a `instant-mcp-server` (using example tools below) and the `instant-mcp-chatbot` configured to use those tools.

Example dialog:



The rest of this file describes these two commands.

## instant-mcp-server

A command-line utility for turning a regular file of Python functions or classes into a full-fledged MCP server.

### Example

Take an existing Python file of functions, such as this file, `my_tools.py`:

```python
# my_tools.py
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
    return f"Welcome to instant-mcp-server, {name}!"
```

Then run the server with your custom tools:

```bash
instant-mcp-server my_tools.py
```

The server will automatically:
- Load all functions from your file (no instant_mcp imports required)
- Convert them to MCP tools
- Generate JSON schemas from your function signatures
- Use your docstrings as tool descriptions

Note: if you just launch the server, it will wait for stdio input. This is designed
to run from inside a system that will dynamically start the server (see below).

### Command-line Options

```
instant-mcp-server [-h] [--transport {stdio,sse}] [--host HOST] [--port PORT] [tools_file]
```

Positional arguments:
  * `tools_file` - Path to the tools file containing functions to serve as MCP tools (default: tools.py)

Options:
  * `-h`, `--help` - show this help message and exit
  * `--transport {stdio,sse}` - Transport method to use (default: `stdio`)
  * `--host HOST` - Host for SSE transport (default: `localhost`)
  * `--port PORT` - Port for SSE transport (default: `8000`)

# Instant MCP Chatbot

A powerful AI chatbot that integrates with Model Context Protocol (MCP) servers and provides observability through Opik tracing. This chatbot can connect to various MCP servers to access specialized tools and capabilities, making it a versatile assistant for different tasks.

## Features

- **MCP Integration**: Connect to multiple Model Context Protocol servers for specialized tool access
- **Opik Observability**: Built-in tracing and observability with Opik integration
- **Interactive Chat Interface**: Rich console interface with command history and auto-completion
- **Python Code Execution**: Execute Python code directly in the chat environment
- **Tool Management**: Discover and use tools from connected MCP servers
- **Configurable**: JSON-based configuration for models and MCP servers
- **Async Support**: Full asynchronous operation for better performance

### MCP Integration

The server implements the full MCP specification:

- **Tool Discovery**: Dynamic tool listing and metadata
- **Tool Execution**: Asynchronous tool calling with proper error handling
- **Protocol Compliance**: Full compatibility with MCP clients
- **Extensibility**: Easy addition of new tools and capabilities

## Example

Create a default configuration file:

```bash
instant-mcp-chatbot --init
```

This creates a `config.json` file with default settings.

Edit `config.json` to specify your model and MCP servers. For example:

```json
{
  "model": "openai/gpt-4o-mini",
  "model_kwargs": {
    "temperature": 0.2
  },
  "mcp_servers": [
    {
      "name": "instant-mcp-server",
      "description": "Instant MCP server from Python files",
      "command": "instant-mcp-server",
      "args": ["/path/to/my_tools.py"]
    }
  ]
}
```

Supported model formats:

- `openai/gpt-4o-mini`
- `anthropic/claude-3-sonnet`
- `google/gemini-pro`
- And many more through LiteLLM

### Basic Commands

Inside the `instant-mcp-chatbot`, you can have a normal LLM conversation.

In addition, you have access to the following meta-commands:

- `/clear` - Clear the conversation history
- `/help` - Show available commands
- `/debug on` or `/debug off` to toggle debug output
- `/show tools` - to list all available tools
- `/show tools SERVER` - to list tools for a specific server
- `/run SERVER.TOOL` - to execute a tool
- `! python_code` - to execute Python code (e.g., '! print(2+2)')
- `quit` or `exit` - Exit the chatbot


### Python Code Execution

Execute Python code by prefixing with `!`:

```
! print("Hello, World!")
! import math
! math.sqrt(16)
```

### Tool Usage

The chatbot automatically discovers and uses tools from connected MCP servers. Simply ask questions that require tool usage, and the chatbot will automatically call the appropriate tools.

## Opik Integration

The chatbot includes built-in Opik observability integration:

### Opik Modes

For the command-line flag `--opik`:

- `hosted` (default): Use hosted Opik service
- `local`: Use local Opik instance
- `disabled`: Disable Opik tracing

### Configure Opik

Set environment variables for Opik:

```bash
# For hosted mode
export OPIK_API_KEY=your_opik_api_key

# For local mode
export OPIK_LOCAL_URL=http://localhost:8080
```

### Command Line Options

```bash
# Use hosted Opik (default)
instant-mcp-chatbot --opik hosted

# Use local Opik
instant-mcp-chatbot --opik local

# Disable Opik
instant-mcp-chatbot --opik disabled
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/comet-ml/instant-mcp)
- **Issues**: [GitHub Issues](https://github.com/comet-ml/instant-mcp/issues)

## Acknowledgments

- Built with [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
- Powered by [LiteLLM](https://github.com/BerriAI/litellm)
- Observability by [Opik](https://opik.ai/)
- Rich console interface by [Rich](https://github.com/Textualize/rich)

## Development

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Format code: `black . && isort .`
6. Commit your changes: `git commit -m "Add feature"`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

### Prerequisites

- Python 3.8 or higher
- OpenAI, Anthropic, or other LLM provider API key (for chatbot functionality)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/comet-ml/instant-mcp.git
cd instant-mcp

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Manually Install Dependencies

```bash
pip install -r requirements.txt
```
