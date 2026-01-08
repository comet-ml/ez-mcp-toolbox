#!/usr/bin/env python3
"""
ez-mcp-optimize: Command-line tool for optimizing LLM applications using Opik.

This tool provides a simple interface to run optimizations on datasets
using Opik's optimization framework.
"""

import argparse
import asyncio
import json
import os
import sys
import signal
import importlib.util
import warnings
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from opik_optimizer.optimization_config.chat_prompt import ChatPrompt
from opik.evaluation import metrics as opik_metrics
from opik import track, opik_context
from rich.console import Console
from dotenv import load_dotenv
import litellm

from .utils import (
    configure_opik,
    init_opik_and_load_dataset,
    resolve_prompt_with_opik,
    load_metrics_by_names_for_optimizer,
    generate_mcp_mermaid_diagram,
    update_opik_span_and_trace_with_usage,
)
from .chatbot import MCPChatbot

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
    model_parameters: Optional[Dict[str, Any]] = None
    config_path: Optional[str] = None
    tools_file: Optional[str] = None
    num: Optional[int] = None
    optimizer: str = "EvolutionaryOptimizer"
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    optimize_kwargs: Optional[Dict[str, Any]] = None


class MCPOptimizer(MCPChatbot):
    """Main optimizer class for running Opik optimizations."""

    def __init__(self, config: EvaluationConfig):
        # Initialize the chatbot with the optimization config parameters
        super().__init__(
            config_path=config.config_path or "ez-config.json",
            system_prompt=config.prompt,
            model_override=config.model,
            model_args_override=config.model_parameters,
            tools_file=config.tools_file,
            debug=config.debug,
        )

        # Store the optimization-specific config
        self.config = config
        self.client: Optional[Any] = None
        self.dataset: Optional[Any] = None

    def configure_opik(self) -> None:
        """Configure Opik based on the specified mode."""
        configure_opik(self.config.opik_mode, "ez-mcp-optimization")

        # Configure manual logging for litellm.completion calls
        if self.config.opik_mode != "disabled":
            try:
                # Store the original completion function
                self._original_completion = litellm.completion

                # Create a manually tracked version that also attaches mermaid diagram
                # Note: _mermaid_diagram will be set later in run_optimization
                @track(name="llm_completion", type="llm")
                def manually_tracked_completion(*args, **kwargs):
                    """Manually tracked version of litellm.completion"""
                    # Call the original completion first
                    resp = self._original_completion(*args, **kwargs)

                    # Extract model from args/kwargs
                    model = kwargs.get("model") or (args[0] if args else None)

                    try:
                        # Add mermaid diagram to trace metadata if available
                        if hasattr(self, "_mermaid_diagram") and self._mermaid_diagram:
                            try:
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

                                existing_metadata["_opik_graph_definition"] = {
                                    "format": "mermaid",
                                    "data": self._mermaid_diagram,
                                }
                                opik_context.update_current_trace(
                                    metadata=existing_metadata
                                )
                            except Exception:
                                pass

                        # Extract token counts and update span with usage information
                        if resp and model:
                            update_opik_span_and_trace_with_usage(model, resp)
                    except Exception:
                        pass

                    return resp

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

    # evaluation task no longer used; removed

    def validate_metrics(self) -> None:
        """Validate that the configured metrics exist before starting async operations.

        This allows for early exit with clean error messages before any async tasks are created.
        """
        # Just validate - don't return the metrics yet
        # This will raise ValueError if metric doesn't exist
        load_metrics_by_names_for_optimizer(
            self.config.metric,
            self.config.metrics_file,
            self.console,
            output_ref=self.config.output_ref,
            reference_field=self.config.reference_field,
        )

    def get_metrics(self) -> List[Any]:
        """Get the metrics to use for optimization.

        Metrics can be loaded from either opik.evaluation.metrics OR --metrics-file.
        Metrics can be either:
        1. A class that when instantiated has a score() method requiring --output mapping
        2. A function that takes (dataset_item, llm_output) as parameters
        """
        return load_metrics_by_names_for_optimizer(
            self.config.metric,
            self.config.metrics_file,
            self.console,
            output_ref=self.config.output_ref,
            reference_field=self.config.reference_field,
        )

    # _load_metrics_from_file removed

    # _list_available_metrics_from_module removed

    @track(name="optimization_run", type="general")
    async def run_optimization(self) -> Any:
        """Run the optimization using Opik."""
        try:
            # Clear messages before optimization to prevent context window overflow
            self.clear_messages()

            # Generate mermaid diagram once at the start
            mermaid_diagram = await generate_mcp_mermaid_diagram(self.mcp_manager)

            # Store mermaid diagram as instance variable for use in manually_tracked_completion
            self._mermaid_diagram = mermaid_diagram

            # Attach mermaid diagram to trace metadata
            # Do this after @track decorator has initialized the trace
            if mermaid_diagram:
                try:
                    from opik import opik_context

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
                    except Exception as e:
                        if self.config.debug:
                            self.console.print(f"⚠️  Could not get current trace: {e}")

                    # Set _opik_graph_definition within metadata
                    existing_metadata["_opik_graph_definition"] = {
                        "format": "mermaid",
                        "data": mermaid_diagram,
                    }
                    opik_context.update_current_trace(metadata=existing_metadata)

                    if self.config.debug:
                        self.console.print(
                            f"✅ Attached mermaid diagram to trace metadata ({len(mermaid_diagram)} chars)"
                        )
                except Exception as e:
                    # Opik not available, continue without tracing
                    if self.config.debug:
                        self.console.print(f"⚠️  Failed to attach mermaid diagram: {e}")
                    pass

            self.console.print("🚀 Starting optimization...")
            self.console.print(f"   - Experiment: {self.config.experiment_name}")
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

            # Get metrics for optimization (these are functions, not class instances)
            metrics = self.get_metrics()
            self.console.print("✅ Metrics loaded successfully!")

            # No need to create evaluation task - optimizer works directly with ChatPrompt

            # Run optimization - Opik handles progress display internally
            self.console.print("🔄 Running optimization...")

            # Use opik_optimizer EvolutionaryOptimizer with optimize_prompt method
            # The optimize_prompt method signature:
            # optimize_prompt(prompt, dataset, metric, experiment_config, n_samples, auto_continue, agent_class, **kwargs)

            # Use the first metric as the primary metric for optimization
            primary_metric_fn = metrics[0] if metrics else None
            if not primary_metric_fn:
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
                            if self.config.debug:
                                self.console.print(
                                    f"🔍 DEBUG: Tool dispatch called for {name} with kwargs: {kwargs}"
                                )
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
                                result = run_async_in_sync_context(
                                    self.mcp_manager.execute_tool_call, tool_call
                                )
                                if self.config.debug:
                                    result_preview = (
                                        str(result)[:100] + "..."
                                        if len(str(result)) > 100
                                        else str(result)
                                    )
                                    self.console.print(
                                        f"🔍 DEBUG: Tool {name} returned: {result_preview}"
                                    )
                                return result
                            except Exception as e:
                                if self.config.debug:
                                    self.console.print(
                                        f"🔍 DEBUG: Tool {name} failed: {e}"
                                    )
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

            # Create ChatPrompt - let it handle message accumulation internally
            # The optimizer will manage clearing between dataset items
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
                "model_parameters" not in optimizer_constructor_kwargs
                and self.config.model_parameters is not None
            ):
                optimizer_constructor_kwargs["model_parameters"] = (
                    self.config.model_parameters
                )

            optimizer = optimizer_class(**optimizer_constructor_kwargs)

            # Prepare parameters for optimizer.optimize_prompt
            # Wrap the metric function to handle message clearing and feedback logging
            # The metric function should take (dataset_item, llm_output) as parameters
            def optimizer_metric(dataset_item, llm_output):
                # Attach mermaid diagram to the current trace (each evaluation gets its own trace)
                # This is called for each evaluation, and each gets its own trace
                if mermaid_diagram:
                    try:
                        from opik import opik_context

                        # Get the current trace - this should be the evaluation task trace
                        current_trace = None
                        existing_metadata = {}
                        try:
                            current_trace = opik_context.get_current_trace()
                            if current_trace:
                                if self.config.debug:
                                    self.console.print(
                                        f"🔍 Found trace in optimizer_metric: {type(current_trace)}"
                                    )
                                if (
                                    hasattr(current_trace, "metadata")
                                    and current_trace.metadata
                                ):
                                    existing_metadata = (
                                        current_trace.metadata.copy()
                                        if isinstance(current_trace.metadata, dict)
                                        else {}
                                    )
                        except Exception as e:
                            if self.config.debug:
                                self.console.print(
                                    f"⚠️  Error getting trace in optimizer_metric: {e}"
                                )

                        # Set _opik_graph_definition within metadata
                        existing_metadata["_opik_graph_definition"] = {
                            "format": "mermaid",
                            "data": mermaid_diagram,
                        }
                        opik_context.update_current_trace(metadata=existing_metadata)

                        if self.config.debug and current_trace:
                            self.console.print(
                                f"✅ Attached mermaid diagram to evaluation trace ({len(mermaid_diagram)} chars)"
                            )
                    except Exception as e:
                        if self.config.debug:
                            self.console.print(
                                f"⚠️  Failed to attach mermaid in optimizer_metric: {e}"
                            )
                        pass

                # Clear messages before each metric evaluation to prevent context window overflow
                # This is the same approach used in the evaluator
                self.clear_messages()

                # Call the metric function directly - it should take (dataset_item, llm_output)
                score = primary_metric_fn(dataset_item, llm_output)

                if self.config.debug:
                    self.console.print(
                        f"🔍 DEBUG: Metric returned score: {score} (type: {type(score)})"
                    )

                # Log feedback score to the current trace for OPIK optimizer
                # This is required for the optimizer to collect feedback scores
                # OPIK optimizers expect feedback_scores to be logged to traces
                try:
                    from opik import opik_context
                    from opik.evaluation.metrics.score_result import ScoreResult

                    # Extract score value and metric name from ScoreResult if that's what we got
                    # Otherwise treat it as a plain float/number
                    if isinstance(score, ScoreResult):
                        score_value = (
                            float(score.value) if score.value is not None else 0.0
                        )
                        metric_name = (
                            score.name
                            if score.name
                            else self.config.metric.split(",")[0].strip()
                        )
                    else:
                        # Plain number or other type
                        score_value = float(score) if score is not None else 0.0
                        metric_name = self.config.metric.split(",")[0].strip()

                    # Try to get the trace object and use log_feedback_score method
                    # This is the recommended way according to OPIK documentation
                    try:
                        current_trace = opik_context.get_current_trace()
                        if current_trace is not None and hasattr(
                            current_trace, "log_feedback_score"
                        ):
                            # Use the trace's log_feedback_score method
                            current_trace.log_feedback_score(
                                name=metric_name,
                                value=score_value,
                            )
                            if self.config.debug:
                                self.console.print(
                                    f"✅ Logged feedback score via trace.log_feedback_score: {metric_name}={score_value}"
                                )
                        else:
                            # Fallback to update_current_trace if trace object method not available
                            from opik.opik_context import FeedbackScoreDict

                            feedback_score: FeedbackScoreDict = {
                                "name": metric_name,
                                "value": score_value,
                            }
                            opik_context.update_current_trace(
                                feedback_scores=[feedback_score]
                            )
                            if self.config.debug:
                                self.console.print(
                                    f"✅ Logged feedback score via update_current_trace: {metric_name}={score_value}"
                                )
                    except AttributeError:
                        # get_current_trace might not exist in some OPIK versions
                        # Fallback to update_current_trace
                        from opik.opik_context import FeedbackScoreDict

                        feedback_score_fallback: FeedbackScoreDict = {
                            "name": metric_name,
                            "value": score_value,
                        }
                        opik_context.update_current_trace(
                            feedback_scores=[feedback_score_fallback]
                        )
                        if self.config.debug:
                            self.console.print(
                                f"✅ Logged feedback score via update_current_trace (fallback): {metric_name}={score_value}"
                            )

                except Exception as e:
                    # If logging fails, continue - the score is still returned
                    # The optimizer will still use the returned score, but may show warnings
                    if self.config.debug:
                        self.console.print(f"⚠️  Failed to log feedback score: {e}")
                        import traceback

                        self.console.print(traceback.format_exc())

                return score

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

            # Update metadata again after optimize_prompt in case it created a new trace context
            if mermaid_diagram:
                try:
                    from opik import opik_context

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
                    opik_context.update_current_trace(metadata=existing_metadata)

                    if self.config.debug:
                        self.console.print(
                            "✅ Re-attached mermaid diagram after optimize_prompt"
                        )
                except Exception:
                    pass

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
        resolved_prompt, prompt_id = self.resolve_prompt(self.config.prompt)
        prompt_display = (
            resolved_prompt[:100] + "..."
            if len(resolved_prompt) > 100
            else resolved_prompt
        )
        self.console.print(f"💬 Prompt: {prompt_display}")
        if prompt_id:
            self.console.print(f"📋 Prompt ID: {prompt_id}")

        self.console.print("\n✅ Optimization completed successfully!")

    async def run(self) -> None:
        """Run the complete optimization process."""
        try:
            # Metrics are validated early in main() before async operations start
            # Configure Opik
            self.configure_opik()

            # Connect to MCP servers using the inherited chatbot method
            self.console.print("🔧 Connecting to MCP servers...")
            await self.connect_all_servers()

            if self.mcp_manager.sessions:
                self.console.print(
                    "✅ MCP connections established - tools will be loaded by MCPChatbot"
                )
                # Get tools for optimization
                tools = await self.mcp_manager._get_all_tools()
                if tools:
                    self.console.print(
                        f"✅ Successfully loaded {len(tools)} tools for optimization"
                    )
                    self._preloaded_tools = tools
                else:
                    self.console.print(
                        "⚠️  No tools returned from MCP server - continuing without tools"
                    )
                    self._preloaded_tools = []
            else:
                self.console.print("❌ No MCP connections available")
                raise RuntimeError("No MCP connections available")

            # Setup client and dataset
            self.setup_client_and_dataset()

            # Run optimization
            results = await self.run_optimization()

            return results

        except asyncio.CancelledError:
            # Handle cancellation gracefully - allow cleanup to run
            self.console.print("⚠️  Optimization was cancelled")
            raise
        except (ValueError, RuntimeError) as e:
            # For validation errors (like unknown metrics), exit cleanly
            # Don't call sys.exit() here - let the exception propagate to main()
            # which will handle it properly
            self.console.print(f"❌ Optimization failed: {e}")
            if self.config.debug:
                import traceback

                self.console.print(traceback.format_exc())
            # Re-raise to let main() handle the exit
            raise
        except Exception as e:
            self.console.print(f"❌ Optimization failed: {e}")
            if self.config.debug:
                import traceback

                self.console.print(traceback.format_exc())
            # Re-raise to let main() handle the exit
            raise
        finally:
            # Restore original litellm.completion if we patched it
            if hasattr(self, "_original_completion"):
                try:
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
                                timeout=30.0,
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
  ez-mcp-optimize --prompt "Answer the question" --dataset "qa-dataset" --metric "LevenshteinRatio" --model-parameters '{"temperature": 0.7, "max_tokens": 1000}'
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
        type=str,
        required=False,
        help="Path to a Python file containing metric definitions. If not provided, metrics will be loaded from opik.evaluation.metrics. Metrics can be either a class with a score() method or a function taking (dataset_item, llm_output).",
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
        help="Number of items to optimize from the dataset (takes first N items, default: all items)",
    )

    parser.add_argument(
        "--optimizer",
        choices=[
            "EvolutionaryOptimizer",
            "FewShotBayesianOptimizer",
            "GepaOptimizer",
            "HierarchicalReflectiveOptimizer",
            "MetaPromptOptimizer",
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

    # Track if we've been interrupted
    interrupted = False
    main_task_ref: Optional[asyncio.Task[Any]] = None

    def signal_handler(signum, frame):
        """Handle SIGINT (Control+C) and SIGTERM gracefully."""
        nonlocal interrupted, main_task_ref
        if not interrupted:
            interrupted = True
            print("\n⚠️  Interrupt received, shutting down gracefully...")

            # Cancel the main task if we have a reference to it
            task_to_cancel = main_task_ref
            if task_to_cancel is not None and not task_to_cancel.done():
                task_to_cancel.cancel()

            # Try to interrupt the event loop if it's running
            try:
                loop = asyncio.get_running_loop()
                # Cancel all tasks - this should cause run_until_complete to exit
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()
            except RuntimeError:
                # No event loop running
                pass
        else:
            # Second interrupt - forcefully terminate
            print("\n⚠️  Second interrupt received, forcefully terminating...")
            os._exit(130)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
        model_parameters=model_parameters,
        config_path=args.config,
        tools_file=args.tools_file,
        num=args.num,
        optimizer=args.optimizer,
        optimizer_kwargs=optimizer_kwargs,
        optimize_kwargs=optimize_kwargs,
    )

    # Create and run optimizer
    optimizer = MCPOptimizer(config)

    # Validate metrics early before starting async operations for clean error exit
    try:
        optimizer.validate_metrics()
    except ValueError as e:
        console = Console()
        console.print(f"❌ {e}")
        sys.exit(1)

    # Handle async execution with proper cleanup
    try:
        import nest_asyncio

        nest_asyncio.apply()

        # Create a custom event loop with proper cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Create the main task and store reference for signal handler
        main_task = loop.create_task(optimizer.run())
        main_task_ref = main_task  # Store reference for signal handler

        exit_code = 0
        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            # Handle KeyboardInterrupt (Control+C)
            if not interrupted:
                print("\n⚠️  Interrupt received, shutting down gracefully...")
            interrupted = True
            exit_code = 130  # Standard exit code for SIGINT
            # Cancel the main task
            if not main_task.done():
                main_task.cancel()
                try:
                    loop.run_until_complete(main_task)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            if not interrupted:
                print("⚠️  Optimization was cancelled")
            exit_code = 130
        except (ValueError, RuntimeError) as e:
            # Handle validation errors cleanly
            print(f"❌ Optimization failed: {e}")
            exit_code = 1
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            if args.debug:
                import traceback

                traceback.print_exc()
            exit_code = 1
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
                                    timeout=30.0,
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
                # Ensure cleanup runs even if interrupted
                if interrupted:
                    print("🧹 Performing cleanup...")
                loop.close()

        # Exit with the appropriate code after cleanup
        if exit_code != 0:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        # Handle KeyboardInterrupt at the outer level
        if not interrupted:
            print("\n⚠️  Interrupt received, shutting down gracefully...")
        sys.exit(130)
    except Exception:
        # Fallback: try to run in a new thread with new event loop
        import concurrent.futures

        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(optimizer.run())
            except KeyboardInterrupt:
                if not interrupted:
                    print("\n⚠️  Interrupt received, shutting down gracefully...")
                return None
            except asyncio.CancelledError:
                # Handle cancellation gracefully
                if not interrupted:
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
                                        timeout=30.0,
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
            try:
                future.result()
            except KeyboardInterrupt:
                if not interrupted:
                    print("\n⚠️  Interrupt received, shutting down gracefully...")
                # Cancel the future
                future.cancel()
                sys.exit(130)


if __name__ == "__main__":
    main()
