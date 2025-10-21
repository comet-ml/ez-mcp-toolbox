#!/usr/bin/env python3
"""
ez-mcp-optimize: Command-line tool for optimizing LLM applications using Opik.

This tool provides a simple interface to run optimizations on datasets
using Opik's optimization framework.
"""

import argparse
import asyncio
import json
import sys
import importlib.util
import warnings
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from opik_optimizer.optimization_config.chat_prompt import ChatPrompt
from opik.evaluation import metrics as opik_metrics
from opik import track
from rich.console import Console
from dotenv import load_dotenv

from .utils import (
    configure_opik,
    init_opik_and_load_dataset,
    resolve_prompt_with_opik,
    load_metrics_by_names,
)
from .mcp_utils import MCPManager, ServerConfig

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


@dataclass
class EvaluationConfig:
    """Configuration for the optimization run."""

    prompt: str
    dataset: str
    metric: str
    metrics_file: Optional[str] = None
    experiment_name: str = "ez-mcp-optimization"
    opik_mode: str = "hosted"
    debug: bool = False
    input_field: str = "input"
    reference_field: str = "answer"
    output_ref: str = "reference"
    # If provided as single FIELD via --output FIELD, use this for ChatPrompt user
    user_field_override: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    model_kwargs: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None
    tools_file: Optional[str] = None
    num: Optional[int] = None
    optimizer: str = "EvolutionaryOptimizer"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    optimize_kwargs: Optional[Dict[str, Any]] = None


class MCPOptimizer:
    """Main optimizer class for running Opik optimizations."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.console = Console()
        self.client: Optional[Any] = None
        self.dataset: Optional[Any] = None
        self.mcp_manager = MCPManager(console=self.console, debug=config.debug)
        self._preloaded_tools: Optional[List[Dict[str, Any]]] = None

    def configure_opik(self) -> None:
        """Configure Opik based on the specified mode."""
        configure_opik(self.config.opik_mode, "ez-mcp-optimization")

        # Configure manual logging for litellm.completion calls
        if self.config.opik_mode != "disabled":
            try:
                import litellm
                from opik import track

                # Store the original completion function
                self._original_completion = litellm.completion

                # Create a manually tracked version
                @track(name="llm_completion", type="llm")
                def manually_tracked_completion(*args, **kwargs):
                    """Manually tracked version of litellm.completion"""
                    return self._original_completion(*args, **kwargs)

                # Replace with our manually tracked version
                litellm.completion = manually_tracked_completion
                self.console.print(
                    "✅ Configured manual tracking for litellm.completion"
                )

            except Exception as e:
                self.console.print(f"⚠️  Failed to configure manual tracking: {e}")

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

    def resolve_prompt(self, prompt_value: str) -> str:
        """Resolve prompt by first checking Opik for a prompt with that name, then fallback to direct value."""
        try:
            if self.client is None:
                raise RuntimeError("Opik client not initialized")
            return resolve_prompt_with_opik(self.client, prompt_value, self.console)

        except Exception as e:
            # If not found or any error, use the prompt value directly
            self.console.print(
                f"⚠️  Prompt '{prompt_value}' not found in Opik ({e}), using as direct prompt"
            )
            return prompt_value

    # evaluation task no longer used; removed

    def get_metrics(self) -> List[Any]:
        """Get the metrics to use for optimization."""
        return load_metrics_by_names(
            self.config.metric, self.config.metrics_file, self.console
        )

    # _load_metrics_from_file removed

    # _list_available_metrics_from_module removed

    @track(name="optimization_run", type="general")
    async def run_optimization(self) -> Any:
        """Run the optimization using Opik."""
        try:
            self.console.print("🚀 Starting optimization...")
            self.console.print(f"   - Experiment: {self.config.experiment_name}")
            self.console.print(f"   - Dataset: {self.config.dataset}")
            self.console.print(f"   - Metric: {self.config.metric}")

            # Resolve the prompt and show it
            resolved_prompt = self.resolve_prompt(self.config.prompt)
            prompt_display = (
                resolved_prompt[:100] + "..."
                if len(resolved_prompt) > 100
                else resolved_prompt
            )
            self.console.print(f"   - Prompt: {prompt_display}")

            # Get metrics for optimization
            metrics = self.get_metrics()
            self.console.print("✅ Metrics loaded successfully!")

            # No need to create evaluation task - optimizer works directly with ChatPrompt

            # Run optimization - Opik handles progress display internally
            self.console.print("🔄 Running optimization...")

            # Use opik_optimizer EvolutionaryOptimizer with optimize_prompt method
            # The optimize_prompt method signature:
            # optimize_prompt(prompt, dataset, metric, experiment_config, n_samples, auto_continue, agent_class, **kwargs)

            # Use the first metric as the primary metric for optimization
            primary_metric = metrics[0] if metrics else None
            if not primary_metric:
                raise ValueError("At least one metric is required for optimization")

            # Prepare MCP-provided tools and a function_map that routes calls back to MCP
            tools_for_prompt = self._preloaded_tools or []

            # Build a function map that forwards tool calls to MCP synchronously
            function_map: Dict[str, Any] = {}

            if tools_for_prompt:
                # Lightweight structures to mimic OpenAI tool call objects
                class _ToolFunction:
                    def __init__(self, name: str, arguments: str):
                        self.name = name
                        self.arguments = arguments

                class _ToolCall:
                    def __init__(self, _id: str, fn: _ToolFunction):
                        self.id = _id
                        self.function = fn

                from .utils import run_async_in_sync_context

                for tool_spec in tools_for_prompt:
                    fn_name = tool_spec.get("function", {}).get("name")
                    if not fn_name:
                        continue

                    def _make_dispatch(name: str):
                        def _tool_dispatch(**kwargs):
                            # Serialize kwargs to JSON string to match expected shape
                            try:
                                args_json = json.dumps(kwargs)
                            except Exception:
                                args_json = "{}"
                            tool_call = _ToolCall(
                                _id="mcp-" + name,
                                fn=_ToolFunction(name=name, arguments=args_json),
                            )
                            # Use synchronous execution that handles async internally
                            # This preserves Opik tracking from execute_tool_call
                            try:
                                return run_async_in_sync_context(
                                    self.mcp_manager.execute_tool_call, tool_call
                                )
                            except Exception as e:
                                return f"Error executing tool '{name}': {e}"

                        return _tool_dispatch

                    function_map[fn_name] = _make_dispatch(fn_name)

            # Create a ChatPrompt object from the resolved prompt
            # Determine which dataset field to use for the user template. If --output was provided
            # as a single FIELD, prefer that; otherwise fall back to --input.
            user_field = (
                self.config.user_field_override
                if self.config.user_field_override
                else self.config.input_field
            )

            chat_prompt = ChatPrompt(
                system=resolved_prompt,
                # Use the selected user field as the user template, e.g., "{question}"
                user="{" + user_field + "}",
                tools=tools_for_prompt if tools_for_prompt else None,
                function_map=function_map if function_map else None,
            )

            # Create the optimizer instance based on config
            optimizer_class = getattr(
                __import__("opik_optimizer", fromlist=[self.config.optimizer]),
                self.config.optimizer,
            )

            # Pass optimizer construction kwargs if provided
            optimizer_constructor_kwargs = self.config.optimizer_kwargs or {}
            # Ensure required constructor args like `model` are present for certain optimizers
            if "model" not in optimizer_constructor_kwargs and self.config.model:
                optimizer_constructor_kwargs["model"] = self.config.model
            if (
                "model_kwargs" not in optimizer_constructor_kwargs
                and self.config.model_kwargs is not None
            ):
                optimizer_constructor_kwargs["model_kwargs"] = self.config.model_kwargs

            optimizer = optimizer_class(**optimizer_constructor_kwargs)

            # Prepare parameters for optimizer.optimize_prompt
            # Wrap the metric in a function matching optimizer's expected signature
            # def optimizer_metric(dataset_item: dict[str, Any], llm_output: str) -> ScoreResult
            def optimizer_metric(dataset_item, llm_output):
                metric_class = primary_metric.__class__
                metric_instance = metric_class()
                # Use the configured reference field from the dataset as the reference input to the metric
                return metric_instance.score(
                    reference=dataset_item[self.config.reference_field],
                    output=llm_output,
                )

            optimize_kwargs = {
                "prompt": chat_prompt,
                "dataset": self.dataset,
                "metric": optimizer_metric,
            }

            # Add n_samples if num is specified
            if self.config.num is not None:
                optimize_kwargs["n_samples"] = self.config.num
                self.console.print(
                    f"📊 Limiting optimization to first {self.config.num} items"
                )

            # Merge in user-provided optimize kwargs
            user_optimize_kwargs = self.config.optimize_kwargs or {}
            optimize_kwargs.update(user_optimize_kwargs)

            eval_results = optimizer.optimize_prompt(**optimize_kwargs)

            self.console.print("✅ Optimization completed!")

            # Display results
            self.display_results(eval_results)

            return eval_results

        except Exception as e:
            self.console.print(f"❌ Optimization failed: {e}")
            raise

    def display_results(self, results: Any) -> None:
        """Display optimization results."""
        self.console.print("\n" + "=" * 60)
        self.console.print("📊 OPTIMIZATION RESULTS", style="bold blue")
        self.console.print("=" * 60)

        # Show optimization information
        self.console.print(f"📈 Experiment: {self.config.experiment_name}")
        self.console.print(f"📊 Dataset: {self.config.dataset}")
        self.console.print(f"🎯 Metric: {self.config.metric}")

        # Show the resolved prompt (truncated if too long)
        resolved_prompt = self.resolve_prompt(self.config.prompt)
        prompt_display = (
            resolved_prompt[:100] + "..."
            if len(resolved_prompt) > 100
            else resolved_prompt
        )
        self.console.print(f"💬 Prompt: {prompt_display}")

        self.console.print("\n✅ Optimization completed successfully!")

    async def run(self) -> None:
        """Run the complete optimization process."""
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

            # Pre-load tools after MCP connections are established
            if self.mcp_manager.sessions:
                self.console.print("🔧 Loading tools for optimization...")
                try:
                    tools = await self.mcp_manager._get_all_tools()
                    if tools:
                        self.console.print(
                            f"✅ Successfully loaded {len(tools)} tools for optimization"
                        )
                        # Store tools for use in optimization
                        self._preloaded_tools = tools
                    else:
                        self.console.print(
                            "⚠️  No tools returned from MCP server - continuing without tools"
                        )
                        self._preloaded_tools = None
                except Exception as e:
                    self.console.print(f"⚠️  Failed to load tools: {e}")
                    self.console.print("⚠️  Continuing without tools")
                    self._preloaded_tools = None
            else:
                self.console.print(
                    "⚠️  No MCP servers configured - continuing without tools"
                )
                self._preloaded_tools = None

            # Setup client and dataset
            self.setup_client_and_dataset()

            # Run optimization
            results = await self.run_optimization()

            return results

        except Exception as e:
            self.console.print(f"❌ Optimization failed: {e}")
            if self.config.debug:
                import traceback

                self.console.print(traceback.format_exc())
            sys.exit(1)
        finally:
            # Restore original litellm.completion if we patched it
            if hasattr(self, "_original_completion"):
                try:
                    import litellm

                    litellm.completion = self._original_completion
                    self.console.print("✅ Restored original litellm.completion")
                except Exception as e:
                    self.console.print(
                        f"⚠️  Failed to restore original litellm.completion: {e}"
                    )

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

            # Wait for any remaining async tasks to complete
            try:
                # Get current event loop
                loop = asyncio.get_running_loop()
                # Get all pending tasks
                pending_tasks = [
                    task for task in asyncio.all_tasks(loop) if not task.done()
                ]
                if pending_tasks:
                    self.console.print(
                        f"⏳ Waiting for {len(pending_tasks)} pending tasks to complete..."
                    )

                    # Filter out LiteLLM service logging tasks specifically
                    litellm_tasks = [
                        task
                        for task in pending_tasks
                        if hasattr(task, "_coro")
                        and "ServiceLogging" in str(task._coro)
                    ]
                    other_tasks = [
                        task for task in pending_tasks if task not in litellm_tasks
                    ]

                    if litellm_tasks:
                        self.console.print(
                            f"🔧 Found {len(litellm_tasks)} LiteLLM service logging tasks"
                        )
                        # Cancel LiteLLM tasks immediately as they're not critical
                        for task in litellm_tasks:
                            try:
                                task.cancel()
                            except Exception:
                                pass

                    # Only wait for non-LiteLLM tasks
                    if other_tasks:
                        try:
                            # Use a more robust approach to avoid recursion
                            done, pending = await asyncio.wait(
                                other_tasks,
                                timeout=2.0,
                                return_when=asyncio.ALL_COMPLETED,
                            )
                            if pending:
                                self.console.print(
                                    f"⚠️  {len(pending)} tasks did not complete within timeout, cancelling..."
                                )
                                # Cancel remaining tasks safely to avoid recursion
                                for task in pending:
                                    try:
                                        task.cancel()
                                    except Exception:
                                        # Ignore cancellation errors to prevent recursion
                                        pass
                        except Exception as wait_error:
                            self.console.print(
                                f"⚠️  Error waiting for tasks: {wait_error}"
                            )
                            # Cancel all tasks if there's an error
                            for task in other_tasks:
                                if not task.done():
                                    try:
                                        task.cancel()
                                    except Exception:
                                        pass
            except Exception as cleanup_error:
                self.console.print(f"⚠️  Error during task cleanup: {cleanup_error}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ez-mcp-optimize: Optimize LLM applications using Opik",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ez-mcp-optimize --prompt "Answer the question" --dataset "my-dataset" --metric "Hallucination"
  ez-mcp-optimize --prompt "Summarize this text" --dataset "summarization-dataset" --metric "LevenshteinRatio" --experiment-name "summarization-test"
  ez-mcp-optimize --prompt "Translate to French" --dataset "translation-dataset" --metric "Hallucination,LevenshteinRatio" --opik local
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --input "question"
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --model-kwargs '{"temperature": 0.7, "max_tokens": 1000}'
  ez-mcp-optimize --prompt "Answer the question" --dataset "large-dataset" --metric "LevenshteinRatio" --num 100
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "CustomMetric" --metrics-file "my_metrics.py"
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --tools-file "my_tools.py"
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --optimizer "FewShotBayesianOptimizer"
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --class-kwargs '{"population_size": 50, "mutation_rate": 0.1}'
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --optimize-kwargs '{"auto_continue": true, "n_samples": 100}'
  ez-mcp-optimize --list-metrics
  ez-mcp-optimize --list-metrics --metrics-file "my_metrics.py"
        """,
    )

    parser.add_argument("--prompt", help="The prompt to use for optimization")

    parser.add_argument("--dataset", help="Name of the dataset to optimize on")

    parser.add_argument(
        "--metric",
        help="Name of the metric(s) to use for optimization. Use comma-separated list for multiple metrics (e.g., 'Hallucination,LevenshteinRatio')",
    )

    parser.add_argument(
        "--metrics-file",
        help="Path to a Python file containing metric definitions. If provided, metrics will be loaded from this file instead of opik.evaluation.metrics",
    )

    parser.add_argument(
        "--experiment-name",
        default="ez-mcp-optimization",
        help="Name for the optimization experiment (default: ez-mcp-optimization)",
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
        help="Output field mapping. Accepts 'REFERENCE=FIELD', 'REFERENCE:FIELD', or just 'FIELD'. If only FIELD is provided, it will be used as the ChatPrompt user field.",
    )

    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metrics and exit",
    )

    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use for optimization (default: gpt-3.5-turbo)",
    )

    parser.add_argument(
        "--model-kwargs",
        type=str,
        help='JSON string of additional keyword arguments to pass to the LLM model (e.g., \'{"temperature": 0.7, "max_tokens": 1000}\')',
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
        help="Number of items to optimize from the dataset (takes first N items, default: all items)",
    )

    parser.add_argument(
        "--optimizer",
        choices=[
            "EvolutionaryOptimizer",
            "FewShotBayesianOptimizer",
            "MetaPromptOptimizer",
            "GepaOptimizer",
            "MiproOptimizer",
        ],
        default="EvolutionaryOptimizer",
        help="Optimizer class to use for optimization (default: EvolutionaryOptimizer)",
    )

    parser.add_argument(
        "--class-kwargs",
        type=str,
        help='JSON string of keyword arguments to pass to the optimizer constructor (e.g., \'{"population_size": 50, "mutation_rate": 0.1}\')',
    )

    parser.add_argument(
        "--optimize-kwargs",
        type=str,
        help='JSON string of keyword arguments to pass to the optimize_prompt() method (e.g., \'{"auto_continue": true, "n_samples": 100}\')',
    )

    return parser.parse_args()


# parse_field_mapping removed


def parse_output_mapping_extended(mapping_str: str) -> tuple[str, str, Optional[str]]:
    """Parse output mapping supporting '=', ':', or single FIELD.

    Returns (output_ref, dataset_field, user_field_override).
    user_field_override is set only for single FIELD form to signal ChatPrompt user template override.
    """
    if "=" in mapping_str or ":" in mapping_str:
        sep = "=" if "=" in mapping_str else ":"
        output_ref, dataset_field = mapping_str.split(sep, 1)
        return output_ref.strip(), dataset_field.strip(), None
    # Single token field name
    field = mapping_str.strip()
    # Default output_ref mirrors evaluator's default
    return "reference", field, field


# validate_input_field removed


# validate_output_mapping removed


def list_available_metrics() -> List[str]:
    """List all available metrics from opik.evaluation.metrics."""
    available_metrics = [
        name
        for name in dir(opik_metrics)
        if not name.startswith("_") and callable(getattr(opik_metrics, name))
    ]
    return sorted(available_metrics)


def main() -> None:
    """Main entry point for ez-mcp-optimize."""
    args = parse_arguments()

    # Handle --list-metrics option
    if args.list_metrics:
        console = Console()

        # Load metrics from file if specified
        if args.metrics_file:
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
            console.print("📊 Available metrics from opik.evaluation.metrics:")
            available_metrics = list_available_metrics()

        for metric in available_metrics:
            console.print(f"   - {metric}")
        return

    # Validate required arguments for optimization
    if not args.prompt:
        console = Console()
        console.print("❌ --prompt is required for optimization")
        sys.exit(1)

    if not args.dataset:
        console = Console()
        console.print("❌ --dataset is required for optimization")
        sys.exit(1)

    if not args.metric:
        console = Console()
        console.print("❌ --metric is required for optimization")
        sys.exit(1)

    # Parse input field mapping only
    input_field = args.input

    # Parse output mapping (REFERENCE=FIELD | REFERENCE:FIELD | FIELD)
    output_ref, reference_field, user_field_override = parse_output_mapping_extended(
        args.output
    )

    # Parse model kwargs JSON
    model_kwargs = None
    if args.model_kwargs:
        try:
            model_kwargs = json.loads(args.model_kwargs)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --model-kwargs: {e}")
            sys.exit(1)

    # Parse class kwargs JSON
    optimizer_kwargs = None
    if args.class_kwargs:
        try:
            optimizer_kwargs = json.loads(args.class_kwargs)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --class-kwargs: {e}")
            sys.exit(1)

    # Parse optimize kwargs JSON
    optimize_kwargs = None
    if args.optimize_kwargs:
        try:
            optimize_kwargs = json.loads(args.optimize_kwargs)
        except json.JSONDecodeError as e:
            console = Console()
            console.print(f"❌ Invalid JSON in --optimize-kwargs: {e}")
            sys.exit(1)

    # Create configuration
    config = EvaluationConfig(
        prompt=args.prompt,
        dataset=args.dataset,
        metric=args.metric,
        metrics_file=args.metrics_file,
        experiment_name=args.experiment_name,
        opik_mode=args.opik,
        debug=args.debug,
        input_field=input_field,
        reference_field=reference_field,
        output_ref=output_ref,
        user_field_override=user_field_override,
        # output mapping no longer needed; metric should accept (dataset_item, llm_output)
        model=args.model,
        model_kwargs=model_kwargs,
        config_path=args.config,
        tools_file=args.tools_file,
        num=args.num,
        optimizer=args.optimizer,
        optimizer_kwargs=optimizer_kwargs,
        optimize_kwargs=optimize_kwargs,
    )

    # Create and run optimizer
    optimizer = MCPOptimizer(config)

    # Handle async execution with proper cleanup
    try:
        import nest_asyncio

        nest_asyncio.apply()

        # Create a custom event loop with proper cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(optimizer.run())
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            print("⚠️  Optimization was cancelled")
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            raise
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

            # Wait for all pending tasks to complete before closing the loop
            try:
                # Get all pending tasks
                pending_tasks = [
                    task for task in asyncio.all_tasks(loop) if not task.done()
                ]
                if pending_tasks:
                    print(
                        f"⏳ Waiting for {len(pending_tasks)} pending tasks to complete..."
                    )

                    # Filter out LiteLLM service logging tasks specifically
                    litellm_tasks = [
                        task
                        for task in pending_tasks
                        if hasattr(task, "_coro")
                        and "ServiceLogging" in str(task._coro)
                    ]
                    other_tasks = [
                        task for task in pending_tasks if task not in litellm_tasks
                    ]

                    if litellm_tasks:
                        print(
                            f"🔧 Found {len(litellm_tasks)} LiteLLM service logging tasks"
                        )
                        # Cancel LiteLLM tasks immediately as they're not critical
                        for task in litellm_tasks:
                            try:
                                task.cancel()
                            except Exception:
                                pass

                    # Only wait for non-LiteLLM tasks
                    if other_tasks:

                        async def wait_for_tasks():
                            try:
                                # Use a more robust approach to avoid recursion
                                done, pending = await asyncio.wait(
                                    other_tasks,
                                    timeout=2.0,
                                    return_when=asyncio.ALL_COMPLETED,
                                )
                                if pending:
                                    print(
                                        f"⚠️  {len(pending)} tasks did not complete within timeout, cancelling..."
                                    )
                                    # Cancel remaining tasks safely to avoid recursion
                                    for task in pending:
                                        try:
                                            task.cancel()
                                        except Exception:
                                            # Ignore cancellation errors to prevent recursion
                                            pass
                            except Exception as wait_error:
                                print(f"⚠️  Error waiting for tasks: {wait_error}")
                                # Cancel all tasks if there's an error
                                for task in other_tasks:
                                    if not task.done():
                                        try:
                                            task.cancel()
                                        except Exception:
                                            pass

                        try:
                            loop.run_until_complete(wait_for_tasks())
                        except Exception as wait_error:
                            print(f"⚠️  Error in task cleanup: {wait_error}")
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
                return new_loop.run_until_complete(optimizer.run())
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                print("⚠️  Optimization was cancelled")
                return None
            except Exception as e:
                print(f"❌ Optimization failed: {e}")
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

                # Wait for all pending tasks to complete before closing the loop
                try:
                    # Get all pending tasks
                    pending_tasks = [
                        task for task in asyncio.all_tasks(new_loop) if not task.done()
                    ]
                    if pending_tasks:
                        print(
                            f"⏳ Waiting for {len(pending_tasks)} pending tasks to complete..."
                        )

                        # Filter out LiteLLM service logging tasks specifically
                        litellm_tasks = [
                            task
                            for task in pending_tasks
                            if hasattr(task, "_coro")
                            and "ServiceLogging" in str(task._coro)
                        ]
                        other_tasks = [
                            task for task in pending_tasks if task not in litellm_tasks
                        ]

                        if litellm_tasks:
                            print(
                                f"🔧 Found {len(litellm_tasks)} LiteLLM service logging tasks"
                            )
                            # Cancel LiteLLM tasks immediately as they're not critical
                            for task in litellm_tasks:
                                try:
                                    task.cancel()
                                except Exception:
                                    pass

                        # Only wait for non-LiteLLM tasks
                        if other_tasks:

                            async def wait_for_tasks():
                                try:
                                    # Use a more robust approach to avoid recursion
                                    done, pending = await asyncio.wait(
                                        other_tasks,
                                        timeout=2.0,
                                        return_when=asyncio.ALL_COMPLETED,
                                    )
                                    if pending:
                                        print(
                                            f"⚠️  {len(pending)} tasks did not complete within timeout, cancelling..."
                                        )
                                        # Cancel remaining tasks safely to avoid recursion
                                        for task in pending:
                                            try:
                                                task.cancel()
                                            except Exception:
                                                # Ignore cancellation errors to prevent recursion
                                                pass
                                except Exception as wait_error:
                                    print(f"⚠️  Error waiting for tasks: {wait_error}")
                                    # Cancel all tasks if there's an error
                                    for task in other_tasks:
                                        if not task.done():
                                            try:
                                                task.cancel()
                                            except Exception:
                                                pass

                            try:
                                new_loop.run_until_complete(wait_for_tasks())
                            except Exception as wait_error:
                                print(f"⚠️  Error in task cleanup: {wait_error}")
                except Exception as e:
                    print(f"⚠️  Error during task cleanup: {e}")
                finally:
                    new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            future.result()


if __name__ == "__main__":
    main()
