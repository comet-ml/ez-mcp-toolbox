#!/usr/bin/env python3
"""
ez-mcp-eval: Command-line tool for evaluating LLM applications using Opik.

This tool provides a simple interface to run evaluations on datasets
using Opik's evaluation framework.
"""

import argparse
import asyncio
import json
import os
import sys
import importlib.util
import warnings
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from opik.evaluation import evaluate
from opik.exceptions import ScoreMethodMissingArguments
from rich.console import Console
from dotenv import load_dotenv

from .utils import (
    configure_opik,
    init_opik_and_load_dataset,
    resolve_prompt_with_opik,
    load_metrics_by_names,
)
from .mcp_utils import ServerConfig
from .chatbot import MCPChatbot


class FieldMappingError(Exception):
    """Custom exception for field mapping errors that should exit cleanly without tracebacks."""

    pass


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

load_dotenv()


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""

    prompt: str
    dataset: str
    metric: str
    metrics_file: Optional[str] = None
    experiment_name: str = "ez-mcp-evaluation-experiment"
    project_name: str = "ez-mcp-evaluation-project"
    opik_mode: str = "hosted"
    debug: bool = False
    input_field: str = "input"
    reference_field: str = "answer"
    output_field: str = "output"
    output_ref: str = "reference"
    model: str = "gpt-3.5-turbo"
    model_parameters: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None
    tools_file: Optional[str] = None
    num: Optional[int] = None


class MCPEvaluator(MCPChatbot):
    """Main evaluator class for running Opik evaluations."""

    def __init__(self, config: EvaluationConfig):
        # Initialize the chatbot with the evaluation config parameters
        super().__init__(
            config_path=config.config_path or "ez-config.json",
            system_prompt=config.prompt,
            model_override=config.model,
            model_args_override=config.model_parameters,
            tools_file=config.tools_file,
            debug=config.debug,
        )

        # Store the evaluation-specific config
        self.config = config
        self.client: Optional[Any] = None
        self.dataset: Optional[Any] = None

    def configure_opik(self) -> None:
        """Configure Opik based on the specified mode."""
        configure_opik(self.config.opik_mode, self.config.project_name)

    def setup_client_and_dataset(self) -> None:
        """Initialize Opik client and load dataset."""
        try:
            self.console.print("🔗 Connecting to Opik...")
            self.client, self.dataset = init_opik_and_load_dataset(
                self.config.dataset, self.console
            )

            self.console.print(f"   - Dataset name: {self.config.dataset}")
            if self.dataset is not None:
                self.console.print(
                    f"   - Items count: {len(self.dataset) if hasattr(self.dataset, '__len__') else 'Unknown'}"
                )

        except Exception as e:
            self.console.print(
                f"❌ Failed to load dataset '{self.config.dataset}': {e}"
            )
            raise

    def resolve_prompt(self, prompt_value: str) -> tuple[str, Optional[str]]:
        """Resolve prompt by first checking Opik for a prompt with that name, then fallback to direct value.

        Returns:
            tuple: (prompt_content, prompt_id) where prompt_id is None if not found in Opik
        """
        try:
            if self.client is None:
                raise RuntimeError("Opik client not initialized")
            return resolve_prompt_with_opik(self.client, prompt_value, self.console)

        except Exception as e:
            # If not found or any error, use the prompt value directly
            self.console.print(
                f"⚠️  Prompt '{prompt_value}' not found in Opik ({e}), using as direct prompt"
            )
            return prompt_value, None

    def create_evaluation_task(self, resolved_prompt: str) -> Any:
        """Create the evaluation task function with pre-loaded tools."""

        def evaluation_task(dataset_item: Any) -> Dict[str, Any]:
            """Synchronous evaluation task that will be called for each dataset item."""
            try:
                if self.config.debug:
                    self.console.print(f"🔍 Processing dataset item: {dataset_item}")

                # Get the input value from the dataset
                input_value = dataset_item.get(self.config.input_field)

                # Tools are now handled by the inherited MCPChatbot class
                if self.config.debug:
                    self.console.print("🔧 Using MCPChatbot's tool handling")

                # Use the inherited chat method from MCPChatbot in quiet mode
                try:
                    # Clear any existing messages and set the system prompt
                    self.messages = [{"role": "system", "content": resolved_prompt}]

                    # Temporarily disable debug output and console for quiet evaluation
                    original_debug = self.debug
                    original_console = self.console
                    original_mcp_debug = self.mcp_manager.debug
                    self.debug = False
                    self.console = None  # Disable console output
                    self.mcp_manager.debug = False  # Disable MCP debug output

                    # Run the async chat method in a sync context using nest_asyncio
                    # to preserve Opik trace context for proper cost logging
                    from .utils import run_async_in_sync_context

                    response = run_async_in_sync_context(self.chat, str(input_value))

                    # Restore original settings
                    self.debug = original_debug
                    self.console = original_console
                    self.mcp_manager.debug = original_mcp_debug

                except Exception as llm_error:
                    self.console.print(f"⚠️  Chat with tools failed: {llm_error}")
                    response = f"Chat Error: {llm_error}"

                if self.config.debug:
                    self.console.print(f"📝 Generated response: {response}")

                # Return simple output structure that can be mapped by scoring_key_mapping
                return {"llm_output": response}

            except Exception as e:
                self.console.print(f"❌ Error processing dataset item: {e}")
                return {"llm_output": f"Error: {e}"}

        return evaluation_task

    def get_metrics(self) -> List[Any]:
        """Get metrics from either opik.evaluation.metrics OR --metrics-file.

        Metrics can be either:
        1. A class that when instantiated has a score() method requiring --output mapping
        2. A function that takes (dataset_item, llm_output) as parameters
        """
        return load_metrics_by_names(
            self.config.metric, self.config.metrics_file, self.console
        )

    def _load_metrics_from_file(self, file_path: str) -> Any:
        from .utils import _load_metrics_from_file as _load

        return _load(file_path, self.console)

    def _list_available_metrics_from_module(self, metrics_module: Any) -> List[str]:
        from .utils import _list_available_metrics_from_module as _list

        return _list(metrics_module)

    async def run_evaluation(self) -> Any:
        """Run the evaluation using Opik."""
        try:
            self.console.print("🚀 Starting evaluation...")
            self.console.print(f"   - Experiment: {self.config.experiment_name}")
            self.console.print(f"   - Project: {self.config.project_name}")
            self.console.print(f"   - Dataset: {self.config.dataset}")
            self.console.print(f"   - Metric: {self.config.metric}")

            # Resolve the prompt and show it
            resolved_prompt, prompt_id = self.resolve_prompt(self.config.prompt)
            prompt_display = (
                resolved_prompt[:100] + "..."
                if len(resolved_prompt) > 100
                else resolved_prompt
            )
            self.console.print(f"   - Prompt: {prompt_display}")
            if prompt_id:
                self.console.print(f"   - Prompt ID: {prompt_id}")

            # Validate input and output fields before proceeding
            self.console.print("🔍 Validating input and output fields...")
            validate_input_field(self.dataset, self.config.input_field, self.console)

            # Get metrics first so we can validate output mapping
            metrics = self.get_metrics()
            validate_output_mapping(
                self.dataset,
                self.config.output_ref,
                self.config.reference_field,
                metrics,
                self.console,
            )

            # Validate that fields have non-empty values
            validate_field_values(
                self.dataset,
                self.config.input_field,
                self.config.reference_field,
                self.console,
            )

            self.console.print("✅ Field validation passed!")

            # Create evaluation task with the already-resolved prompt
            evaluation_task = self.create_evaluation_task(resolved_prompt)

            # Run evaluation - let Opik handle its own progress display
            # Opik's evaluate function has built-in tqdm progress bar
            self.console.print("🔄 Running evaluation...")

            # Use synchronous evaluation task with single threading to avoid async/sync conflicts
            scoring_key_mapping = {
                "output": "llm_output",  # metric parameter "output" -> task output "llm_output"
                self.config.output_ref: self.config.reference_field,  # metric parameter "reference" -> dataset field "output"
            }

            if self.config.debug:
                self.console.print(f"🔍 Scoring key mapping: {scoring_key_mapping}")
                self.console.print(f"   output_ref: {self.config.output_ref}")
                self.console.print(f"   reference_field: {self.config.reference_field}")

            eval_kwargs = {
                "experiment_name": self.config.experiment_name,
                "dataset": self.dataset,
                "task": evaluation_task,
                "scoring_metrics": metrics,
                "scoring_key_mapping": scoring_key_mapping,
                "task_threads": 1,  # Disable multi-threading to avoid async/sync conflicts
            }

            # Add nb_samples if num is specified
            if self.config.num is not None:
                eval_kwargs["nb_samples"] = self.config.num
                self.console.print(
                    f"📊 Limiting evaluation to first {self.config.num} items"
                )

            eval_results = evaluate(**eval_kwargs)

            self.console.print("✅ Evaluation completed!")

            # Display results
            self.display_results(eval_results)

            return eval_results

        except ScoreMethodMissingArguments as e:
            # Catch Opik's specific error and provide clearer guidance
            error_msg = str(e)
            self.console.print("\n❌ Evaluation failed: Field mapping error")

            # Extract available fields from error message
            available_fields = []
            missing_args = []
            mismapped_keys = []

            # Parse error message to extract useful information
            import re

            if "missing arguments:" in error_msg:
                # Extract missing arguments
                match = re.search(r"missing arguments:\s*\[(.*?)\]", error_msg)
                if match:
                    missing_args = [
                        arg.strip().strip("'\"") for arg in match.group(1).split(",")
                    ]

            if "available keys" in error_msg:
                # Extract available keys
                match = re.search(
                    r"available keys.*?are:\s*\[(.*?)\]", error_msg, re.IGNORECASE
                )
                if match:
                    available_fields = [
                        key.strip().strip("'\"") for key in match.group(1).split(",")
                    ]

            if "didn't match anything" in error_msg:
                # Extract mismapped keys
                match = re.search(r"didn't match anything:\s*\[(.*?)\]", error_msg)
                if match:
                    mismapped_keys = [
                        key.strip().strip("'\"") for key in match.group(1).split(",")
                    ]

            # If we couldn't extract from error, try to get from dataset
            if not available_fields:
                available_fields = get_available_dataset_fields(self.dataset)

            # Display helpful error message
            self.console.print(f"   {error_msg}")

            if missing_args:
                self.console.print(
                    f"\n   Missing required arguments: {', '.join(missing_args)}"
                )

            if available_fields:
                self.console.print(
                    f"\n   Available dataset fields: {', '.join(available_fields)}"
                )

            if mismapped_keys:
                self.console.print(
                    f"\n   Field mapping issue: '{mismapped_keys[0]}' was not found in the dataset"
                )
                # Suggest corrections
                if available_fields:
                    suggestions = [
                        f
                        for f in available_fields
                        if any(
                            x in f.lower()
                            for x in [
                                mismapped_keys[0].lower(),
                                "answer",
                                "reference",
                                "expected",
                                "target",
                                "label",
                                "output",
                            ]
                        )
                    ]
                    if suggestions:
                        self.console.print(f"   💡 Did you mean: {suggestions[0]}?")

            # Get metric parameters and suggest fix
            try:
                metrics = self.get_metrics()
                for metric in metrics:
                    import inspect

                    if hasattr(metric, "score"):
                        sig = inspect.signature(metric.score)
                        param_names = [p for p in sig.parameters.keys() if p != "self"]
                        if param_names:
                            self.console.print(
                                f"\n   Metric '{metric.__class__.__name__}' expects parameters: {', '.join(param_names)}"
                            )

                            # Suggest a fix
                            if param_names and available_fields:
                                # Find reference parameter (usually first or named 'reference')
                                ref_param = "reference"
                                if "reference" in param_names:
                                    ref_param = "reference"
                                elif len(param_names) > 0:
                                    ref_param = param_names[0]

                                # Try to find a likely reference field
                                likely_ref_fields = [
                                    f
                                    for f in available_fields
                                    if any(
                                        x in f.lower()
                                        for x in [
                                            "answer",
                                            "reference",
                                            "expected",
                                            "target",
                                            "label",
                                            "expected_output",
                                        ]
                                    )
                                ]
                                suggested_ref = (
                                    likely_ref_fields[0]
                                    if likely_ref_fields
                                    else (
                                        available_fields[0]
                                        if available_fields
                                        else "FIELD_NAME"
                                    )
                                )

                                self.console.print("\n   💡 Suggested fix:")
                                self.console.print(
                                    f"      --output {ref_param}={suggested_ref}"
                                )

                                # Verify input field
                                likely_input_fields = [
                                    f
                                    for f in available_fields
                                    if any(
                                        x in f.lower()
                                        for x in [
                                            "input",
                                            "question",
                                            "query",
                                            "prompt",
                                        ]
                                    )
                                ]
                                if (
                                    likely_input_fields
                                    and self.config.input_field not in available_fields
                                ):
                                    self.console.print(
                                        f"      --input {likely_input_fields[0]}"
                                    )
                            break
            except Exception:
                pass

            self.console.print("")
            # Raise a custom exception that will be caught and handled gracefully
            raise FieldMappingError()

        except FieldMappingError:
            # Field mapping errors already have clear error messages printed
            # Just re-raise to exit cleanly
            raise
        except ValueError:
            # Re-raise ValueError (these are our validation/field mapping errors)
            # They already have clear error messages, so just pass through
            raise
        except Exception as e:
            error_msg = str(e)
            # Check if this is a field-related error
            if (
                "not found" in error_msg.lower()
                or "missing" in error_msg.lower()
                or "key" in error_msg.lower()
            ):
                available_fields = get_available_dataset_fields(self.dataset)
                if available_fields:
                    self.console.print(
                        f"\n   Available dataset fields: {', '.join(available_fields)}"
                    )
            self.console.print(f"❌ Evaluation failed: {error_msg}")
            raise

    def display_results(self, results: Any) -> None:
        """Display evaluation results."""
        self.console.print("\n" + "=" * 60)
        self.console.print("📊 EVALUATION RESULTS", style="bold blue")
        self.console.print("=" * 60)

        # Show evaluation information
        self.console.print(f"📈 Experiment: {self.config.experiment_name}")
        self.console.print(f"📈 Project: {self.config.project_name}")
        self.console.print(f"📊 Dataset: {self.config.dataset}")
        self.console.print(f"🎯 Metric: {self.config.metric}")

        # Show the resolved prompt (truncated if too long)
        resolved_prompt, prompt_id = self.resolve_prompt(self.config.prompt)
        prompt_display = (
            resolved_prompt[:100] + "..."
            if len(resolved_prompt) > 100
            else resolved_prompt
        )
        self.console.print(f"💬 Prompt: {prompt_display}")
        if prompt_id:
            self.console.print(f"📋 Prompt ID: {prompt_id}")

        self.console.print("\n✅ Evaluation completed successfully!")

    async def run(self) -> None:
        """Run the complete evaluation process."""
        try:
            # Configure Opik
            self.configure_opik()

            # Load MCP configuration if provided
            if self.config.tools_file:
                # Resolve tools file path (download from URL if necessary)
                from .utils import resolve_tools_file_path

                resolved_tools_file = resolve_tools_file_path(
                    self.config.tools_file, self.console
                )

                # Create dynamic MCP server configuration using tools file
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
            elif self.config.config_path:
                servers = self.mcp_manager.load_mcp_config(self.config.config_path)
                if servers:
                    self.console.print(f"📡 Loaded {len(servers)} MCP server(s)")
                    await self.mcp_manager.connect_all_servers(servers)
                else:
                    self.console.print("⚠️  No MCP servers configured")

            # Connect to MCP servers using the inherited chatbot method
            self.console.print("🔧 Connecting to MCP servers...")
            await self.connect_all_servers()

            if self.mcp_manager.sessions:
                self.console.print(
                    "✅ MCP connections established - tools will be loaded by MCPChatbot"
                )
            else:
                self.console.print("❌ No MCP connections available")
                raise RuntimeError("No MCP connections available")

            # Setup client and dataset
            self.setup_client_and_dataset()

            # Run evaluation
            results = await self.run_evaluation()

            return results

        except (FieldMappingError, ValueError) as e:
            # These exceptions already have clear error messages printed
            # Suppress traceback by setting exception context to None
            # FieldMappingError is raised after printing helpful error messages
            # ValueError is raised from validation functions that also print helpful messages
            if self.config.debug:
                import traceback

                self.console.print(traceback.format_exc())
            # Clear exception context to prevent traceback printing
            e.__context__ = None
            e.__cause__ = None
            raise  # Re-raise to exit cleanly
        except Exception as e:
            self.console.print(f"❌ Evaluation failed: {e}")
            if self.config.debug:
                import traceback

                self.console.print(traceback.format_exc())
            raise
        finally:
            # Clean up MCP connections
            if self.mcp_manager.sessions:
                await self.mcp_manager.close()

            # Properly clean up LiteLLM async clients first
            try:
                from litellm.llms.custom_httpx.async_client_cleanup import (
                    close_litellm_async_clients,
                )

                if asyncio.iscoroutinefunction(close_litellm_async_clients):
                    await close_litellm_async_clients()
                else:
                    close_litellm_async_clients()
            except Exception as cleanup_error:
                self.console.print(
                    f"⚠️  Error cleaning up LiteLLM clients: {cleanup_error}"
                )

            # Simplified cleanup - just close MCP connections and let the rest handle itself
            try:
                # Get current event loop
                loop = asyncio.get_running_loop()
                # Get all pending tasks
                pending_tasks = [
                    task for task in asyncio.all_tasks(loop) if not task.done()
                ]
                if pending_tasks:
                    # Cancel all tasks except the current one to avoid hanging
                    current_task = asyncio.current_task(loop)
                    for task in pending_tasks:
                        if task != current_task:
                            try:
                                task.cancel()
                                # Suppress exceptions from cancelled tasks to avoid "Task exception was never retrieved" warnings
                                task.exception()
                            except Exception:
                                pass

            except Exception:
                # Suppress cleanup errors to avoid cluttering output
                pass


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ez-mcp-eval: Evaluate LLM applications using Opik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ez-mcp-eval --prompt "Answer the question" --dataset "my-dataset" --metric "Hallucination" --metrics-file "my_metrics.py"
  ez-mcp-eval --prompt "Summarize this text" --dataset "summarization-dataset" --metric "LevenshteinRatio" --metrics-file "my_metrics.py" --experiment-name "summarization-test"
  ez-mcp-eval --prompt "Translate to French" --dataset "translation-dataset" --metric "Hallucination,LevenshteinRatio" --metrics-file "my_metrics.py" --opik local
  ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --metrics-file "my_metrics.py" --input "question" --output "reference=answer"
  ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --metrics-file "my_metrics.py" --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'
  ez-mcp-eval --prompt "Answer the question" --dataset "large-dataset" --metric "LevenshteinRatio" --metrics-file "my_metrics.py" --num 100
  ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "CustomMetric" --metrics-file "my_metrics.py"
  ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --metrics-file "my_metrics.py" --tools-file "my_tools.py"
  ez-mcp-eval --list-metrics --metrics-file "my_metrics.py"
  ez-mcp-eval --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio"
        """,
    )

    parser.add_argument("--prompt", help="The prompt to use for evaluation")

    parser.add_argument("--dataset", help="Name of the dataset to evaluate on")

    parser.add_argument(
        "--metric",
        help="Name of the metric(s) to use for evaluation. Use comma-separated list for multiple metrics (e.g., 'Hallucination,LevenshteinRatio')",
    )

    parser.add_argument(
        "--metrics-file",
        type=str,
        required=False,
        help="Path to a Python file containing metric definitions. If not provided, metrics will be loaded from opik.evaluation.metrics.",
    )

    parser.add_argument(
        "--experiment-name",
        default="ez-mcp-evaluation-experiment",
        help="Name for the evaluation experiment (default: ez-mcp-evaluation-experiment)",
    )

    parser.add_argument(
        "--project-name",
        default="ez-mcp-evaluation-project",
        help="Name for the evaluation project (default: ez-mcp-evaluation-project)",
    )

    parser.add_argument(
        "--opik",
        choices=["local", "hosted", "disabled"],
        default="hosted",
        help="Opik tracing mode: local, hosted, or disabled (default: hosted)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    parser.add_argument(
        "--input",
        default="input",
        help="Input field name in the dataset (default: input)",
    )

    parser.add_argument(
        "--output",
        default="reference=answer",
        help="Output field mapping in format reference=DATASET_FIELD (default: reference=answer)",
    )

    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics and exit",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use for evaluation (default: from config file or openai/gpt-3.5-turbo)",
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

    parser.add_argument(
        "--config",
        type=str,
        help="Path to MCP server configuration file (default: ez-config.json)",
    )

    parser.add_argument(
        "--tools-file",
        type=str,
        help="Path to a Python file containing tool definitions, or URL to download the file from. If provided, will create an MCP server configuration using this file.",
    )

    parser.add_argument(
        "--num",
        type=int,
        help="Number of items to evaluate from the dataset (takes first N items, default: all items)",
    )

    return parser.parse_args()


def parse_field_mapping(mapping_str: str) -> tuple[str, str]:
    """Parse field mapping in format REFERENCE=QUESTION."""
    if "=" not in mapping_str:
        raise ValueError(
            f"Field mapping must be in format REFERENCE=QUESTION, got: {mapping_str}"
        )

    reference, question = mapping_str.split("=", 1)
    return reference.strip(), question.strip()


def parse_output_mapping(mapping_str: str) -> tuple[str, str]:
    """Parse output mapping in format reference=DATASET_FIELD."""
    if "=" not in mapping_str:
        raise ValueError(
            f"Output mapping must be in format reference=DATASET_FIELD, got: {mapping_str}"
        )

    reference, dataset_field = mapping_str.split("=", 1)
    return reference.strip(), dataset_field.strip()


def get_sample_dataset_item(dataset: Any) -> Optional[Dict[str, Any]]:
    """Get a sample item from a dataset to inspect available fields.

    Works with:
    - Opik Dataset objects (by iterating)
    - Regular iterable datasets
    - Lists/dicts

    Returns:
        A dictionary representing one dataset item, or None if unable to extract.
    """
    if not dataset:
        return None

    try:
        # Try iteration first (works for Opik datasets and most iterables)
        try:
            # Create iterator and get first item
            iterator = iter(dataset)
            first_item = next(iterator)
            if isinstance(first_item, dict):
                return first_item
        except (StopIteration, TypeError, AttributeError):
            pass

        # Try direct indexing
        if hasattr(dataset, "__getitem__"):
            try:
                first_item = dataset[0]
                if isinstance(first_item, dict):
                    return first_item
            except (IndexError, KeyError, TypeError):
                pass

        # If dataset itself is a dict, return it
        if isinstance(dataset, dict):
            return dataset

    except Exception:
        # If any error occurs, return None
        pass

    return None


def get_available_dataset_fields(dataset: Any) -> List[str]:
    """Get list of available field names from a dataset.

    Returns:
        List of field names, or empty list if unable to determine.
    """
    sample_item = get_sample_dataset_item(dataset)
    if sample_item and isinstance(sample_item, dict):
        return list(sample_item.keys())
    return []


def validate_field_values(
    dataset: Any,
    input_field: str,
    reference_field: str,
    console: Console,
) -> None:
    """Validate that input and reference fields have non-empty values in dataset items."""
    if not dataset:
        return

    try:
        empty_input_items = []
        empty_reference_items = []
        items_checked = 0
        max_check = 10  # Check first 10 items

        # Try to iterate through dataset items
        try:
            iterator = iter(dataset)
            for item in iterator:
                items_checked += 1
                if items_checked > max_check:
                    break

                if not isinstance(item, dict):
                    continue

                # Check input field
                input_value = item.get(input_field)
                if input_value is None or (
                    isinstance(input_value, str) and not input_value.strip()
                ):
                    empty_input_items.append(items_checked)

                # Check reference field
                ref_value = item.get(reference_field)
                if ref_value is None or (
                    isinstance(ref_value, str) and not ref_value.strip()
                ):
                    empty_reference_items.append(items_checked)

        except (TypeError, AttributeError):
            # Dataset is not directly iterable, skip validation
            console.print("⚠️  Cannot iterate dataset - skipping value validation")
            return

        if empty_input_items or empty_reference_items:
            error_parts = []
            if empty_input_items:
                error_parts.append(
                    f"Input field '{input_field}' has empty/null values in items: {empty_input_items[:5]}"
                    + (
                        f" (and {len(empty_input_items) - 5} more)"
                        if len(empty_input_items) > 5
                        else ""
                    )
                )
            if empty_reference_items:
                error_parts.append(
                    f"Reference field '{reference_field}' has empty/null values in items: {empty_reference_items[:5]}"
                    + (
                        f" (and {len(empty_reference_items) - 5} more)"
                        if len(empty_reference_items) > 5
                        else ""
                    )
                )

            console.print("\n❌ Dataset validation failed:")
            for part in error_parts:
                console.print(f"   {part}")
            console.print(
                "\n   💡 Dataset fields must have non-empty values. Please check and fix corrupted dataset entries."
            )
            raise ValueError(
                f"Dataset contains empty/null values: {', '.join(error_parts)}"
            )

    except ValueError:
        # Re-raise ValueError (our validation errors)
        raise
    except Exception as e:
        console.print(f"⚠️  Could not validate field values: {e}")
        # Don't fail validation if we can't check values


def validate_input_field(dataset: Any, input_field: str, console: Console) -> None:
    """Validate that the input field exists in the dataset items."""
    if not dataset:
        console.print("❌ Dataset is empty, cannot validate input field")
        raise ValueError("Dataset is empty, cannot validate input field")

    # Try to get a sample item to check fields
    try:
        first_item = get_sample_dataset_item(dataset)

        if first_item is None:
            console.print(
                "⚠️  Cannot inspect dataset structure - skipping field validation"
            )
            console.print("   Field validation will be performed during evaluation")
            return

        # first_item is guaranteed to be Dict[str, Any] here due to get_sample_dataset_item return type
        # Use assert for runtime safety - mypy understands this better than isinstance check
        assert isinstance(first_item, dict), "Dataset items must be dictionaries"

        if input_field not in first_item:
            available_fields = list(first_item.keys())
            console.print(f"\n❌ Input field '{input_field}' not found in dataset")
            console.print(f"   Available fields: {', '.join(available_fields)}")

            # Suggest similar field names
            suggestions = [
                f
                for f in available_fields
                if input_field.lower() in f.lower() or f.lower() in input_field.lower()
            ]
            if suggestions:
                console.print(f"   💡 Did you mean one of: {', '.join(suggestions)}?")

            console.print(
                f"\n   Example usage: --input {available_fields[0] if available_fields else 'FIELD_NAME'}"
            )
            raise ValueError(
                f"Input field '{input_field}' not found in dataset. Available fields: {', '.join(available_fields)}"
            )

    except ValueError:
        # Re-raise ValueError (our validation errors)
        raise
    except Exception as e:
        console.print(f"⚠️  Could not validate input field: {e}")
        console.print("   Field validation will be performed during evaluation")
        # Don't fail validation if we can't inspect the dataset


def validate_output_mapping(
    dataset: Any,
    output_ref: str,
    reference_field: str,
    metrics: List[Any],
    console: Console,
) -> None:
    """Validate that the output mapping is correct."""
    if not dataset:
        console.print("❌ Dataset is empty, cannot validate output mapping")
        raise ValueError("Dataset is empty, cannot validate output mapping")

    # Try to get a sample item to check fields
    try:
        first_item = get_sample_dataset_item(dataset)

        if first_item is not None and isinstance(first_item, dict):
            # Check that reference_field exists in dataset
            if reference_field not in first_item:
                available_fields = list(first_item.keys())
                console.print(
                    f"\n❌ Reference field '{reference_field}' not found in dataset"
                )
                console.print(f"   Available fields: {', '.join(available_fields)}")

                # Suggest similar field names
                suggestions = [
                    f
                    for f in available_fields
                    if reference_field.lower() in f.lower()
                    or f.lower() in reference_field.lower()
                ]
                if suggestions:
                    console.print(
                        f"   💡 Did you mean one of: {', '.join(suggestions)}?"
                    )

                # Suggest a command fix
                if available_fields:
                    # Try to find a field that looks like a reference/answer
                    likely_fields = [
                        f
                        for f in available_fields
                        if any(
                            x in f.lower()
                            for x in [
                                "answer",
                                "reference",
                                "expected",
                                "output",
                                "target",
                                "label",
                            ]
                        )
                    ]
                    suggested_field = (
                        likely_fields[0] if likely_fields else available_fields[0]
                    )
                    console.print(
                        f"\n   Example usage: --output {output_ref}={suggested_field}"
                    )

                raise ValueError(
                    f"Reference field '{reference_field}' not found in dataset. Available fields: {', '.join(available_fields)}"
                )
        else:
            # Can't inspect dataset structure, but we'll still validate metrics
            console.print(
                "⚠️  Cannot inspect dataset structure - skipping field validation"
            )
            console.print("   Field validation will be performed during evaluation")

        # Always validate metric parameters regardless of dataset type
        for metric in metrics:
            # Get the metric's score method signature to check parameters
            import inspect

            try:
                # Skip validation for FunctionMetricAdapter (function metrics)
                # They use **kwargs and don't need --output mapping
                if metric.__class__.__name__ == "FunctionMetricAdapter":
                    continue

                # Try to find the score method
                if hasattr(metric, "score"):
                    sig = inspect.signature(metric.score)
                elif hasattr(metric, "__call__"):
                    # If no score method, check the __call__ method
                    sig = inspect.signature(metric.__call__)
                else:
                    console.print(
                        f"⚠️  Metric '{metric.__class__.__name__}' has no score or __call__ method, skipping parameter validation"
                    )
                    continue

                param_names = list(sig.parameters.keys())
                # Remove 'self' parameter
                if "self" in param_names:
                    param_names.remove("self")

                # Check if this metric uses **kwargs (function-style, doesn't need --output)
                # If so, skip validation since function metrics don't need --output mapping
                has_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in sig.parameters.values()
                )
                if has_kwargs:
                    # This is a function metric - skip validation
                    continue

                # Check if the metric expects the output_ref parameter
                # The output_ref should match one of the metric's parameters
                if output_ref not in param_names:
                    console.print(
                        f"\n❌ Output reference '{output_ref}' is not a valid parameter for metric '{metric.__class__.__name__}'"
                    )
                    console.print(
                        f"   Available metric parameters: {', '.join(param_names)}"
                    )

                    # Suggest the correct parameter name
                    if param_names:
                        likely_params = [
                            p
                            for p in param_names
                            if "ref" in p.lower()
                            or "expected" in p.lower()
                            or "target" in p.lower()
                        ]
                        if likely_params:
                            console.print(f"   💡 Did you mean: {likely_params[0]}?")
                            console.print(
                                f"\n   Example usage: --output {likely_params[0]}=FIELD_NAME"
                            )

                    raise ValueError(
                        f"Output reference '{output_ref}' is not a valid parameter for metric '{metric.__class__.__name__}'. "
                        f"Available parameters: {', '.join(param_names)}"
                    )
            except ValueError:
                # Re-raise ValueError from our validation
                raise
            except Exception as e:
                console.print(
                    f"⚠️  Could not validate metric parameters for {metric.__class__.__name__}: {e}"
                )
                # Continue with other metrics

    except ValueError:
        # Re-raise ValueError (our validation errors)
        raise
    except Exception as e:
        console.print(f"⚠️  Could not validate output mapping: {e}")
        console.print("   Field validation will be performed during evaluation")
        # Don't fail validation if we can't inspect the dataset


def main() -> None:
    """Main entry point for ez-mcp-eval."""
    args = parse_arguments()

    # Handle --list-metrics option
    if args.list_metrics:
        console = Console()

        if args.metrics_file:
            # List metrics from the metrics file
            try:
                # Load the module from file
                spec = importlib.util.spec_from_file_location(
                    "metric_module", args.metrics_file
                )
                if spec is None or spec.loader is None:
                    console.print(f"❌ Could not load module from {args.metrics_file}")
                    sys.exit(1)

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                console.print(f"📊 Available metrics from {args.metrics_file}:")
                available_metrics = [
                    name
                    for name in dir(module)
                    if not name.startswith("_") and callable(getattr(module, name))
                ]
                available_metrics = sorted(available_metrics)
            except Exception as e:
                console.print(
                    f"❌ Error loading metrics from file {args.metrics_file}: {e}"
                )
                sys.exit(1)
        else:
            # List metrics from opik.evaluation.metrics
            from opik.evaluation import metrics as opik_metrics
            from .utils import _list_available_metrics_from_module

            console.print("📊 Available metrics from opik.evaluation.metrics:")
            available_metrics = _list_available_metrics_from_module(opik_metrics)

        for metric in available_metrics:
            console.print(f"   - {metric}")
        return

    # Validate required arguments for evaluation
    if not args.prompt:
        console = Console()
        console.print("❌ --prompt is required for evaluation")
        sys.exit(1)

    if not args.dataset:
        console = Console()
        console.print("❌ --dataset is required for evaluation")
        sys.exit(1)

    if not args.metric:
        console = Console()
        console.print("❌ --metric is required for evaluation")
        sys.exit(1)

    # Parse field mappings
    input_field = args.input
    output_ref, reference_field = parse_output_mapping(args.output)

    # Load model from config file if --model not provided, similar to chatbot behavior
    config_path = args.config or "ez-config.json"
    model_from_config = None
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
                model_from_config = config_data.get("model")
    except Exception:
        pass  # If config loading fails, use default

    # Use args.model if provided, otherwise use model from config, otherwise use default
    model_to_use = args.model or model_from_config or "openai/gpt-3.5-turbo"

    # Parse model parameters JSON (with backwards compatibility for model-kwargs)
    model_parameters = None
    if args.model_parameters and args.model_kwargs:
        warnings.warn(
            "Both --model-parameters and --model-kwargs were provided. "
            "Using --model-parameters and ignoring --model-kwargs.",
            UserWarning,
            stacklevel=2,
        )
    if args.model_parameters:
        try:
            model_parameters = json.loads(args.model_parameters)
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
            model_parameters = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --model-kwargs: {e}")
            sys.exit(1)

    # Create configuration
    config = EvaluationConfig(
        prompt=args.prompt,
        dataset=args.dataset,
        metric=args.metric,
        metrics_file=args.metrics_file,
        experiment_name=args.experiment_name,
        project_name=args.project_name,
        opik_mode=args.opik,
        debug=args.debug,
        input_field=input_field,
        reference_field=reference_field,
        output_field="llm_output",  # Task always returns {"llm_output": response}
        output_ref=output_ref,  # The metric's expected field name (e.g., "reference")
        model=model_to_use,
        model_parameters=model_parameters,
        config_path=args.config,
        tools_file=args.tools_file,
        num=args.num,
    )

    # Create and run evaluator
    evaluator = MCPEvaluator(config)

    # Handle async execution with proper cleanup
    try:
        import nest_asyncio

        nest_asyncio.apply()

        # Create a custom event loop with proper cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(evaluator.run())
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            print("⚠️  Evaluation was cancelled")
            sys.exit(1)
        except (FieldMappingError, ValueError):
            # These exceptions already have clear error messages printed
            # Exit cleanly without printing traceback
            sys.exit(1)
        except Exception as e:
            # For other exceptions, print error message
            print(f"❌ Evaluation failed: {e}")
            if evaluator.config.debug:
                import traceback

                traceback.print_exc()
            sys.exit(1)
        finally:
            # Properly clean up LiteLLM async clients first
            try:
                from litellm.llms.custom_httpx.async_client_cleanup import (
                    close_litellm_async_clients,
                )

                if asyncio.iscoroutinefunction(close_litellm_async_clients):
                    loop.run_until_complete(close_litellm_async_clients())
                else:
                    close_litellm_async_clients()
            except Exception as cleanup_error:
                print(f"⚠️  Error cleaning up LiteLLM clients: {cleanup_error}")

            # Simplified cleanup - just cancel remaining tasks and close loop
            try:
                # Get all pending tasks
                pending_tasks = [
                    task for task in asyncio.all_tasks(loop) if not task.done()
                ]
                if pending_tasks:
                    print(f"⏳ Found {len(pending_tasks)} pending tasks, cancelling...")
                    for task in pending_tasks:
                        try:
                            task.cancel()
                        except Exception:
                            pass
                    print("✅ Cleanup completed")
            except Exception as e:
                print(f"⚠️  Error during task cleanup: {e}")
            finally:
                loop.close()

    except Exception:
        # Fallback: try to run in a new thread with new event loop
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(evaluator.run())
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                print("⚠️  Evaluation was cancelled")
                return None
            except Exception as e:
                print(f"❌ Evaluation failed: {e}")
                raise
            finally:
                # Properly clean up LiteLLM async clients first
                try:
                    from litellm.llms.custom_httpx.async_client_cleanup import (
                        close_litellm_async_clients,
                    )

                    if asyncio.iscoroutinefunction(close_litellm_async_clients):
                        new_loop.run_until_complete(close_litellm_async_clients())
                    else:
                        close_litellm_async_clients()
                except Exception as cleanup_error:
                    print(f"⚠️  Error cleaning up LiteLLM clients: {cleanup_error}")

                # Simplified cleanup - just cancel remaining tasks and close loop
                try:
                    # Get all pending tasks
                    pending_tasks = [
                        task for task in asyncio.all_tasks(new_loop) if not task.done()
                    ]
                    if pending_tasks:
                        print(
                            f"⏳ Found {len(pending_tasks)} pending tasks, cancelling..."
                        )
                        for task in pending_tasks:
                            try:
                                task.cancel()
                            except Exception:
                                pass
                        print("✅ Cleanup completed")
                except Exception as e:
                    print(f"⚠️  Error during task cleanup: {e}")
                finally:
                    new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            future.result()


if __name__ == "__main__":
    main()
