# EZ MCP Toolbox

*A Comet ML Open Source Project*

This Python toolbox contains four command-line easy to use utilities:

1. `ez-mcp-server` - turns a file of Python functions into a MCP server
2. `ez-mcp-chatbot` - interactively debug MCP servers, with traces logged to [Opik](https://www.comet.com/site/products/opik/)
3. `ez-mcp-eval` - evaluate LLM applications using Opik's evaluation framework
4. `ez-mcp-optimize` - optimize LLM applications using Opik's optimization framework

## Why?

The `ez-mcp-server` allows a quick way to examine tools, signatures, descriptions, latency, and return values. Combined with the chatbot, you can create a fast workflow to interate on your MCP tools.

The `ez-mcp-chatbot` allows a quick method to examine and debug LLM and MCP tool interactions, with observability available through [Opik](https://github.com/comet-ml/opik). Although the [Opik Playground](https://www.comet.com/docs/opik/opik-university/prompt-engineering/prompt-playground) gives you the ability to test your prompts on datasets, do A/B testing, and more, this chatbot gives you a command-line interaction, debugging tools, combined with Opik observability.

The `ez-mcp-eval` and `ez-mcp-optimize` commands provide evaluation and optimization capabilities for your LLM applications, enabling you to measure performance and automatically improve your prompts using Opik's evaluation and optimization frameworks.

## Installation

```
pip install ez-mcp-toolbox --upgrade
```

## Quick start

### Interactive Chat with MCP Tools
```
ez-mcp-chatbot
```

That will start a `ez-mcp-server` (using example tools below) and the `ez-mcp-chatbot` configured to use those tools.

### Evaluate LLM Applications
```
ez-mcp-eval --prompt "Answer the question" --dataset "my-dataset" --metric "Hallucination" --output "reference=answer"
```

This will evaluate your LLM application using Opik's evaluation framework with your dataset and chosen metrics. The `--output` parameter is required for class metrics (like built-in Opik metrics) to map metric parameters to dataset fields.

You can also limit the evaluation to the first N items of the dataset:

```bash
ez-mcp-eval --prompt "Answer the question" --dataset "large-dataset" --metric "Hallucination" --num 100
```

### Customize the chatbot

You can customize the chatbot's behavior with a custom system prompt:

```bash
# Use a custom system prompt
ez-mcp-chatbot --prompt "You are a helpful coding assistant"

# Create a default configuration
ez-mcp-chatbot --init
```

Example dialog:

![ez-mcp-video](https://github.com/user-attachments/assets/296d7084-becd-467c-878c-16daec714b65)

This interaction of the LLM with the MCP tools will be logged, and available for examination and debugging in Opik:

<img width="800" alt="chatbot interaction as logged to opik" src="https://github.com/user-attachments/assets/3ad0d79a-7f99-4211-aede-5e0cd81d80c3" />

The rest of this file describes these three commands.

## ez-mcp-server

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
    return f"Welcome to ez-mcp-server, {name}!"
```

Then run the server with your custom tools:

```bash
ez-mcp-server my_tools.py
```

You can also load tools from installed Python modules:

```bash
ez-mcp-server opik_optimizer.utils.core
```

Or download tools from a URL:

```bash
ez-mcp-server https://example.com/my_tools.py
```

The server will automatically:
- Load all functions from your file or module (no ez_mcp_toolbox imports required)
- Convert them to MCP tools
- Generate JSON schemas from your function signatures
- Use your docstrings as tool descriptions

Note: if you just launch the server, it will wait for stdio input. This is designed
to run from inside a system that will dynamically start the server (see below).

### Command-line Options

```
ez-mcp-server [-h] [--transport {stdio,sse}] [--host HOST] [--port PORT] [--include INCLUDE] [--exclude EXCLUDE] [--quiet] [tools_file]
```

Positional arguments:
  * `tools_file` - Path to tools file, module name, URL to download from, or 'none' to disable tools (e.g., 'my_tools.py', 'opik_optimizer.utils.core', 'https://example.com/tools.py', or 'none') (default: DEMO)

Options:
  * `-h`, `--help` - show this help message and exit
  * `--transport {stdio,sse}` - Transport method to use (default: `stdio`)
  * `--host HOST` - Host for SSE transport (default: `localhost`)
  * `--port PORT` - Port for SSE transport (default: `8000`)
  * `--include INCLUDE` - Python regex pattern to include only matching tool names
  * `--exclude EXCLUDE` - Python regex pattern to exclude matching tool names
  * `--quiet` - Suppress all output messages

### Tool Filtering

You can control which tools are loaded using the `--include` and `--exclude` flags with Python regex patterns:

```bash
# Include only tools with "add" or "multiply" in the name
ez-mcp-server my_tools.py --include "add|multiply"

# Exclude tools with "greet" or "time" in the name
ez-mcp-server my_tools.py --exclude "greet|time"

# Use both filters together
ez-mcp-server my_tools.py --include ".*number.*" --exclude ".*square.*"

# Use with default tools
ez-mcp-server --include "add" --exclude "greet"
```

**Filtering Logic:**
- The `--include` filter is applied first, keeping only tools whose names match the regex pattern
- The `--exclude` filter is then applied, removing any tools whose names match the regex pattern
- Both filters can be used together for fine-grained control
- Invalid regex patterns will cause the server to exit with an error message

# Ez MCP Chatbot

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
ez-mcp-chatbot --init
```

This creates a `ez-config.json` file with default settings.

Edit `ez-config.json` to specify your model and MCP servers. For example:

```json
{
  "model": "openai/gpt-4o-mini",
  "model_parameters": {
    "temperature": 0.2
  },
  "mcp_servers": [
    {
      "name": "ez-mcp-server",
      "description": "Ez MCP server from Python files",
      "command": "ez-mcp-server",
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

Inside the `ez-mcp-chatbot`, you can have a normal LLM conversation.

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
! print(self.messages)
! import math
! math.sqrt(16)
```

### Tool Usage

The chatbot automatically discovers and uses tools from connected MCP servers. Simply ask questions that require tool usage, and the chatbot will automatically call the appropriate tools.

## System Prompts

The chatbot uses a system prompt to define its behavior and personality. You can customize this using the `--prompt` command line option, which supports:
- Direct strings: `--prompt "You are a helpful assistant"`
- File paths: `--prompt ./my_prompt.txt`
- Opik prompt names: `--prompt my_optimized_prompt`

### Default System Prompt

By default, the chatbot uses this system prompt:

```
You are a helpful AI system for answering questions that can be answered
with any of the available tools.
```

### Custom System Prompts

You can override the default system prompt to customize the chatbot's behavior:

```bash
# Direct string prompts
ez-mcp-chatbot --prompt "You are an expert Python developer who helps with coding tasks."
ez-mcp-chatbot --prompt "You are a data scientist who specializes in analyzing datasets and creating visualizations."
ez-mcp-chatbot --prompt "You are a friendly AI assistant who loves to help users with their questions and tasks."

# Load prompt from file
ez-mcp-chatbot --prompt ./my_custom_prompt.txt

# Load prompt from Opik (if you have optimized prompts stored there)
ez-mcp-chatbot --prompt my_optimized_coding_assistant
```

The system prompt affects how the chatbot:
- Interprets user requests
- Decides which tools to use
- Structures its responses
- Maintains conversation context

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
ez-mcp-chatbot --opik hosted

# Use local Opik
ez-mcp-chatbot --opik local

# Disable Opik
ez-mcp-chatbot --opik disabled

# Use custom system prompt
ez-mcp-chatbot --prompt "You are a helpful coding assistant"

# Combine options
ez-mcp-chatbot --prompt "You are a data analysis expert" --opik local --debug

# Use custom tools file
ez-mcp-chatbot --tools-file "my_tools.py"

# Use tools file from URL
ez-mcp-chatbot --tools-file "https://example.com/my_tools.py"

# Override model arguments
ez-mcp-chatbot --model-args '{"temperature": 0.7, "max_tokens": 1000}'

# Override both model and model arguments
ez-mcp-chatbot --model "openai/gpt-4" --model-args '{"temperature": 0.3, "max_tokens": 2000}'
```

#### Available Options

- `config_path` - Path to the configuration file (default: ez-config.json)
- `--opik {local,hosted,disabled}` - Opik tracing mode (default: hosted)
- `--init` - Create a default ez-config.json file and exit
- `--debug` - Enable debug output during processing
- `--prompt TEXT` - Custom system prompt for the chatbot (overrides default)
- `--model MODEL` - Override the model specified in the config file
- `--tools-file TOOLS_FILE` - Path to a Python file containing tool definitions, or URL to download the file from. If provided, will create an MCP server configuration using this file.
- `--model-args MODEL_ARGS` - JSON string of additional keyword arguments to pass to the LLM model

## ez-mcp-eval

A command-line utility for evaluating LLM applications using Opik's evaluation framework. This tool provides a simple interface to run evaluations on datasets with various metrics, enabling you to measure and improve your LLM application's performance.

### Features

- **Dataset Evaluation**: Run evaluations on your datasets using Opik's evaluation framework
- **Multiple Metrics**: Support for various evaluation metrics (Hallucination, LevenshteinRatio, etc.)
- **Opik Integration**: Full integration with Opik for observability and tracking
- **Flexible Configuration**: Customizable prompts, models, and evaluation parameters
- **Rich Output**: Beautiful console output with progress tracking and results display

### Basic Usage

```bash
# Using built-in Opik metric (class metric - requires --output)
ez-mcp-eval --prompt "Answer the question" --dataset "my-dataset" --metric "Hallucination" --output "reference=answer"

# Using function metric from file (no --output needed)
ez-mcp-eval --prompt "Answer the question" --dataset "my-dataset" --metric "my_metric" --metrics-file "my_metrics.py"
```

### Command-line Options

```
ez-mcp-eval [-h] [--prompt PROMPT] [--dataset DATASET] [--metric METRIC]
            [--metrics-file METRICS_FILE] [--experiment-name EXPERIMENT_NAME]
            [--project-name PROJECT_NAME] [--opik {local,hosted,disabled}] [--debug] [--input INPUT]
            [--output OUTPUT] [--list-metrics] [--model MODEL]
            [--model-parameters MODEL_PARAMETERS] [--config CONFIG] [--tools-file TOOLS_FILE]
            [--num NUM]
```

#### Required Arguments

- `--prompt PROMPT` - The prompt to use for evaluation
- `--dataset DATASET` - Name of the dataset to evaluate on
- `--metric METRIC` - Name of the metric(s) to use for evaluation (comma-separated for multiple)

#### Optional Arguments

- `--metrics-file METRICS_FILE` - Path to a Python file containing metric definitions. If not provided, metrics will be loaded from `opik.evaluation.metrics`. If a metric name exists in both, the metrics file takes precedence.
- `--experiment-name EXPERIMENT_NAME` - Name for the evaluation experiment (default: ez-mcp-evaluation-experiment)
- `--project-name PROJECT_NAME` - Name for the evaluation project (default: ez-mcp-evaluation-project)
- `--opik {local,hosted,disabled}` - Opik tracing mode (default: hosted)
- `--debug` - Enable debug output
- `--input INPUT` - Input field name in the dataset (default: input)
- `--output OUTPUT` - Output field mapping in format reference=DATASET_FIELD (default: reference=answer). **Required only for class metrics** with a `score()` method. Function metrics don't need this parameter as they receive the full dataset item.
- `--list-metrics` - List all available metrics and exit
- `--model MODEL` - LLM model to use for evaluation (default: gpt-3.5-turbo)
- `--model-parameters MODEL_PARAMETERS` - JSON string of additional keyword arguments to pass to the LLM model (e.g., '{"temperature": 0.7, "max_tokens": 1000}')
- `--config CONFIG` - Path to MCP server configuration file (default: ez-config.json)
- `--tools-file TOOLS_FILE` - Path to a Python file containing tool definitions, or URL to download the file from. If provided, will create an MCP server configuration using this file.
- `--num NUM` - Number of items to evaluate from the dataset (takes first N items, default: all items)

### Dataset Loading

The `ez-mcp-eval` command supports loading datasets from two sources:

1. **Opik datasets**: If the dataset exists in your Opik account, it will be loaded directly
2. **opik_optimizer.datasets**: If the dataset is not found in Opik, the tool will automatically check for a function with the same name in `opik_optimizer.datasets` and create the dataset using that function

This allows you to use both pre-existing Opik datasets and dynamically generated datasets from the `opik_optimizer` package.

### Examples

#### Basic Evaluation
```bash
# Simple evaluation with Hallucination metric
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination"
```

#### Multiple Metrics
```bash
# Evaluate with multiple metrics
ez-mcp-eval --prompt "Summarize this text" --dataset "summarization-dataset" --metric "Hallucination,LevenshteinRatio"
```

#### Custom Experiment Name
```bash
# Use a custom experiment name
ez-mcp-eval --prompt "Translate to French" --dataset "translation-dataset" --metric "LevenshteinRatio" --experiment-name "french-translation-test"
```

#### Custom Model and Parameters
```bash
# Use a different model with custom parameters
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --model "gpt-4" --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'
```

#### Using opik_optimizer Datasets
```bash
# Use a dataset from opik_optimizer.datasets (automatically created if not in Opik)
ez-mcp-eval --prompt "Answer the question" --dataset "my_optimizer_dataset" --metric "Hallucination"
```

#### Custom Field Mappings
```bash
# Custom input and output field mappings
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --input "question" --output "reference=answer"
```

### Field Validation

The `ez-mcp-eval` command now includes automatic validation of input and output field mappings to prevent common configuration errors:

#### Input Field Validation
- **What it checks**: The `--input` field must exist in the dataset items
- **When it runs**: Before starting the evaluation
- **Error handling**: If the field doesn't exist, the command stops with a clear error message showing available fields

#### Output Field Validation
- **What it checks** (only for class metrics with `score()` method):
  - The `--output` VALUE (dataset field) must exist in the dataset items
  - The `--output` KEY (metric parameter) must be a valid parameter for the selected metric(s) score method
- **When it runs**: Before starting the evaluation
- **Error handling**: If validation fails, the command stops with clear error messages
- **Note**: Function metrics skip this validation as they don't require `--output`

#### Example Validation Errors

```bash
# Input field not found in dataset
❌ Input field 'question' not found in dataset items
   Available fields: input, answer

# Output field not found in dataset
❌ Reference field 'response' not found in dataset items
   Available fields: input, answer

# Invalid metric parameter
❌ Output reference 'reference' is not a valid parameter for metric 'LevenshteinRatio' score method
   Available parameters: output, reference
```

This validation helps catch configuration errors early, saving time and preventing failed evaluations.

#### Using Custom Metrics from File
```bash
# Use custom metrics defined in a Python file
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "CustomMetric" --metrics-file "my_metrics.py"
```

#### Using Custom Tools File
```bash
# Use a custom tools file for MCP server configuration
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --tools-file "my_tools.py"

# Use tools file from URL
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --tools-file "https://example.com/my_tools.py"
```

#### List Available Metrics
```bash
# See all available metrics
ez-mcp-eval --list-metrics
```

#### Debug Mode
```bash
# Enable debug output for troubleshooting
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination" --debug
```

### Metrics System

The `ez-mcp-eval` command supports two types of metrics that can be loaded from two sources:

#### Metric Types

1. **Class Metrics**: Classes that when instantiated have a `score()` method requiring `--output` mapping
   - Example: Built-in Opik metrics like `Hallucination`, `LevenshteinRatio`
   - These use Opik's scoring framework with field mappings
   - **Require `--output` parameter** to map metric parameters to dataset fields

2. **Function Metrics**: Functions that take `(dataset_item, output)` as parameters
   - Simple Python functions that receive the full dataset item and LLM output
   - Must use `output` as the parameter name for the LLM output
   - Access fields like `reference` from `dataset_item` (e.g., `dataset_item.get("reference")`)
   - **Do NOT require `--output` parameter**

#### Metric Sources

Metrics can be loaded from:
- `opik.evaluation.metrics` (built-in Opik metrics) - used when `--metrics-file` is not provided
- `--metrics-file` (custom Python file) - if provided, metrics from this file take precedence over built-in metrics with the same name

#### Example Class Metric (Built-in)

```bash
# Use built-in Opik metric (class with score() method)
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination" --output "reference=answer"
```

#### Example Function Metric (Custom)

Create a custom metric file `my_metrics.py`:

```python
# my_metrics.py
def custom_similarity(dataset_item, output):
    """
    Custom metric function that takes dataset_item and output.
    
    The function should use 'output' as the parameter name for the LLM output,
    and access fields like 'reference' from dataset_item.
    
    Args:
        dataset_item: Full dictionary of the dataset item (e.g., {"input": "...", "answer": "...", "reference": "..."})
        output: The LLM's output string
        
    Returns:
        Score (float) or ScoreResult object
    """
    # Access fields from dataset_item - commonly 'reference', 'answer', etc.
    reference = dataset_item.get("reference") or dataset_item.get("answer", "")
    
    # Your custom evaluation logic here
    # Calculate similarity, score, etc. using 'output' and 'reference'
    similarity = len(set(output.split()) & set(reference.split())) / len(set(reference.split()))
    
    return similarity
```

Then use it:
```bash
# Function metric - no --output needed!
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "custom_similarity" --metrics-file "my_metrics.py"
```

#### Example Class Metric (Custom)

You can also define custom class metrics:

```python
# my_metrics.py
from opik.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def score(self, output, reference):
        # Your custom evaluation logic here
        # Return a score
        return 0.8  # Example score
```

Then use it with `--output`:
```bash
ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "CustomMetric" --metrics-file "my_metrics.py" --output "reference=answer"
```

#### When to Use Each Type

- **Use Class Metrics** when:
  - You want to use Opik's built-in metrics
  - You need the full Opik scoring framework integration
  - You're familiar with Opik's metric interface
  
- **Use Function Metrics** when:
  - You want simpler, more direct metric implementation
  - You need direct access to the full dataset item
  - You don't need field mapping configuration

### Opik Integration

The `ez-mcp-eval` tool integrates seamlessly with Opik for:

- **Dataset Management**: Load datasets from your Opik workspace
- **Prompt Management**: Use prompts stored in Opik or provide direct text
- **Experiment Tracking**: Track evaluation experiments with custom names
- **Observability**: Full tracing of LLM calls and evaluation processes

### Environment Setup

For Opik integration, set up your environment:

```bash
# For hosted Opik
export OPIK_API_KEY=your_opik_api_key

# For local Opik
export OPIK_LOCAL_URL=http://localhost:8080
```

### Available Metrics

The tool supports all metrics available in Opik's evaluation framework. Use `--list-metrics` to see the complete list, which includes:

- **Hallucination**: Detect hallucinated content in responses
- **LevenshteinRatio**: Measure text similarity using Levenshtein distance
- **ExactMatch**: Check for exact string matches
- **F1Score**: Calculate F1 score for classification tasks
- And many more...

### Output

The tool provides rich console output including:

- Progress tracking during evaluation
- Dataset information and statistics
- Evaluation results and metrics
- Error handling and debugging information
- Integration with Opik's experiment tracking

## ez-mcp-optimize

A command-line utility for optimizing LLM applications using Opik's optimization framework. This tool provides a simple interface to run prompt optimization on datasets with various metrics and optimizers, enabling you to improve your LLM application's performance through automated optimization.

### Features

- **Prompt Optimization**: Run optimization on your prompts using Opik's optimization framework
- **Multiple Optimizers**: Support for various optimization algorithms (EvolutionaryOptimizer, FewShotBayesianOptimizer, etc.)
- **Opik Integration**: Full integration with Opik for observability and tracking
- **Flexible Configuration**: Customizable prompts, models, and optimization parameters
- **Rich Output**: Beautiful console output with progress tracking and results display

### Basic Usage

```bash
# Using built-in Opik metric (class metric - requires --output)
ez-mcp-optimize --prompt "Answer the question" --dataset "my-dataset" --metric "Hallucination" --output "reference=answer"

# Using function metric from file (no --output needed)
ez-mcp-optimize --prompt "Answer the question" --dataset "my-dataset" --metric "my_metric" --metrics-file "my_metrics.py"
```

### Command-line Options

```
ez-mcp-optimize [-h] [--prompt PROMPT] [--dataset DATASET] [--metric METRIC]
                --metrics-file METRICS_FILE [--experiment-name EXPERIMENT_NAME]
                [--opik {local,hosted,disabled}] [--debug] [--input INPUT]
                [--output OUTPUT] [--list-metrics] [--model MODEL]
                [--model-parameters MODEL_PARAMETERS] [--config CONFIG] [--tools-file TOOLS_FILE]
                [--num NUM] [--optimizer OPTIMIZER] [--class-kwargs CLASS_KWARGS]
                [--optimize-kwargs OPTIMIZE_KWARGS]
```

#### Required Arguments

- `--prompt PROMPT` - The prompt to use for optimization
- `--dataset DATASET` - Name of the dataset to optimize on
- `--metric METRIC` - Name of the metric(s) to use for optimization (comma-separated for multiple)

#### Optional Arguments

- `--metrics-file METRICS_FILE` - Path to a Python file containing metric definitions. If not provided, metrics will be loaded from `opik.evaluation.metrics`. If a metric name exists in both, the metrics file takes precedence. Metrics can be either class metrics (with `score()` method) or function metrics (taking `dataset_item, output` where `output` is the LLM output and `reference` can be accessed from `dataset_item`).
- `--experiment-name EXPERIMENT_NAME` - Name for the optimization experiment (default: ez-mcp-optimization)
- `--opik {local,hosted,disabled}` - Opik tracing mode (default: hosted)
- `--debug` - Enable debug output
- `--input INPUT` - Input field name in the dataset (default: input)
- `--output OUTPUT` - Output field mapping. Accepts 'REFERENCE=FIELD', 'REFERENCE:FIELD', or just 'FIELD'. If only FIELD is provided, it will be used as the ChatPrompt user field. (default: reference=answer). **Required only for class metrics** with a `score()` method. Function metrics don't need this parameter as they receive the full dataset item.
- `--list-metrics` - List all available metrics and exit
- `--model MODEL` - LLM model to use for optimization (default: gpt-3.5-turbo)
- `--model-parameters MODEL_PARAMETERS` - JSON string of additional keyword arguments to pass to the LLM model (e.g., '{"temperature": 0.7, "max_tokens": 1000}')
- `--config CONFIG` - Path to MCP server configuration file (default: ez-config.json)
- `--tools-file TOOLS_FILE` - Path to a Python file containing tool definitions, or URL to download the file from. If provided, will create an MCP server configuration using this file.
- `--num NUM` - Number of items to optimize from the dataset (takes first N items, default: all items)
- `--optimizer OPTIMIZER` - Optimizer class to use for optimization (default: EvolutionaryOptimizer)
- `--class-kwargs CLASS_KWARGS` - JSON string of keyword arguments to pass to the optimizer constructor
- `--optimize-kwargs OPTIMIZE_KWARGS` - JSON string of keyword arguments to pass to the optimize_prompt() method

### Available Optimizers

The tool supports various optimization algorithms:

- **EvolutionaryOptimizer** (default): Genetic algorithm-based optimization
- **FewShotBayesianOptimizer**: Bayesian optimization with few-shot examples
- **MetaPromptOptimizer**: Meta-learning based optimization
- **GepaOptimizer**: Gradient-based optimization
- **HierarchicalReflectiveOptimizer**: Hierarchical reflection-based optimization

### Examples

#### Basic Optimization
```bash
# Simple optimization with built-in Opik metric (class metric)
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination" --output "reference=answer"

# Or with custom function metric from file
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "hallucination_function" --metrics-file "my_metrics.py"
```

#### Multiple Metrics
```bash
# Optimize with multiple built-in metrics
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination,LevenshteinRatio" --output "reference=answer"

# Or with custom metrics from file
ez-mcp-optimize --prompt "Summarize this text" --dataset "summarization-dataset" --metric "hallucination_function,levenshtein_ratio_function" --metrics-file "my_metrics.py"
```

#### Custom Optimizer
```bash
# Use a different optimizer with built-in metric
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --output "reference=answer" --optimizer "FewShotBayesianOptimizer"

# Or with custom metric
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "levenshtein_ratio_function" --metrics-file "my_metrics.py" --optimizer "FewShotBayesianOptimizer"
```

#### Custom Model and Parameters
```bash
# Use a different model with built-in metric
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --output "reference=answer" --model "gpt-4" --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'

# Or with custom metric
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "levenshtein_ratio_function" --metrics-file "my_metrics.py" --model "gpt-4" --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'
```

#### Custom Optimizer Parameters
```bash
# Use custom optimizer parameters
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --output "reference=answer" --class-kwargs '{"population_size": 50, "mutation_rate": 0.1}'
```

#### Custom Optimization Parameters
```bash
# Use custom optimization parameters
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --output "reference=answer" --optimize-kwargs '{"auto_continue": true, "n_samples": 100}'
```

### Metrics System

The `ez-mcp-optimize` command supports the same metrics system as `ez-mcp-eval`. Metrics can be loaded from `opik.evaluation.metrics` or from a custom `--metrics-file`, and can be either class metrics or function metrics.

#### Quick Reference

- **Class Metrics**: Require `--output` parameter (e.g., `--output "reference=answer"`)
- **Function Metrics**: Do NOT require `--output` parameter
- **Metric Sources**: `opik.evaluation.metrics` (default) or `--metrics-file` (overrides built-in with same name)
- **`--metrics-file`**: Optional - use built-in metrics if not provided

#### Example: Using Built-in Class Metrics

```bash
# No --metrics-file needed! Uses opik.evaluation.metrics
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "Hallucination" --output "reference=answer"
```

#### Example: Using Custom Function Metrics

Create `my_metrics.py`:

```python
# my_metrics.py
def my_optimization_metric(dataset_item, output):
    """
    Custom metric for optimization.
    
    The function should use 'output' as the parameter name for the LLM output,
    and access fields like 'reference' from dataset_item.
    
    Args:
        dataset_item: Full dictionary of the dataset item (e.g., {"input": "...", "reference": "..."})
        output: The LLM's output string
        
    Returns:
        Score (float) - higher is better for optimization
    """
    # Access 'reference' (or other fields) from dataset_item
    reference = dataset_item.get("reference") or dataset_item.get("answer", "")
    # Your optimization logic here using 'output' and 'reference'
    return calculate_score(output, reference)
```

Use it without `--output`:
```bash
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "my_optimization_metric" --metrics-file "my_metrics.py"
```

#### Example: Using Custom Class Metrics

```python
# my_metrics.py
from opik.evaluation.metrics import BaseMetric

class MyClassMetric(BaseMetric):
    def score(self, output, reference):
        # Your scoring logic
        return calculate_score(output, reference)
```

Use with `--output`:
```bash
ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "MyClassMetric" --metrics-file "my_metrics.py" --output "reference=answer"
```

### Opik Integration

The `ez-mcp-optimize` tool integrates seamlessly with Opik for:

- **Dataset Management**: Load datasets from your Opik workspace
- **Prompt Management**: Use prompts stored in Opik or provide direct text
- **Experiment Tracking**: Track optimization experiments with custom names
- **Observability**: Full tracing of LLM calls and optimization processes

### Environment Setup

For Opik integration, set up your environment:

```bash
# For hosted Opik
export OPIK_API_KEY=your_opik_api_key

# For local Opik
export OPIK_LOCAL_URL=http://localhost:8080
```

### Output

The tool provides rich console output including:

- Progress tracking during optimization
- Dataset information and statistics
- Optimization results and metrics
- Error handling and debugging information
- Integration with Opik's experiment tracking

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [GitHub Repository](https://github.com/comet-ml/ez-mcp-toolbox)
- **Issues**: [GitHub Issues](https://github.com/comet-ml/ez-mcp-toolbox/issues)

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
git clone https://github.com/comet-ml/ez-mcp-toolbox.git
cd ez-mcp-toolbox

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Manually Install Dependencies

```bash
pip install -r requirements.txt
```
