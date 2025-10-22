#!/usr/bin/env python3
"""
Unit tests for ez_mcp_toolbox.evaluator module.
Tests the validation functionality for input and output fields.
"""

import pytest
from unittest.mock import Mock, patch
from rich.console import Console
import tempfile
import os

# Import the evaluator module
from ez_mcp_toolbox.evaluator import (
    validate_input_field,
    validate_output_mapping,
    parse_output_mapping,
    MCPEvaluator,
)


class TestFieldValidation:
    """Test cases for field validation functionality."""

    def test_validate_input_field_success(self):
        """Test successful input field validation."""
        # Mock dataset with valid field
        dataset = [{"input": "test input", "answer": "test answer"}]
        console = Console()

        # Should not raise an exception
        validate_input_field(dataset, "input", console)

    def test_validate_input_field_missing_field(self):
        """Test input field validation with missing field."""
        # Mock dataset without the specified field
        dataset = [{"question": "test question", "answer": "test answer"}]
        console = Console()

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Input field 'input' not found in dataset"
        ):
            validate_input_field(dataset, "input", console)

    def test_validate_input_field_empty_dataset(self):
        """Test input field validation with empty dataset."""

        # Mock empty dataset (iterator that yields nothing)
        class EmptyDataset:
            def __iter__(self):
                return iter([])

        dataset = EmptyDataset()
        console = Console()

        # Should not raise an exception (returns early)
        validate_input_field(dataset, "input", console)

    def test_validate_input_field_non_dict_items(self):
        """Test input field validation with non-dictionary items."""
        # Mock dataset with non-dict items
        dataset = ["not a dict", "also not a dict"]
        console = Console()

        # Should not raise an exception (returns early)
        validate_input_field(dataset, "input", console)

    def test_validate_output_mapping_success(self):
        """Test successful output mapping validation."""
        # Mock dataset with valid fields
        dataset = [{"input": "test input", "answer": "test answer"}]
        console = Console()

        # Mock metric with valid score method parameter
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "TestMetric"
        mock_metric.score = Mock()

        # Mock the signature to include 'reference' parameter
        from unittest.mock import Mock as MockSignature

        mock_sig = MockSignature()
        mock_sig.parameters = {
            "self": Mock(),
            "reference": Mock(),
            "other_param": Mock(),
        }

        with patch("inspect.signature", return_value=mock_sig):
            # Should not raise an exception
            validate_output_mapping(
                dataset, "reference", "answer", [mock_metric], console
            )

    def test_validate_output_mapping_missing_reference_field(self):
        """Test output mapping validation with missing reference field."""
        # Mock dataset without the reference field
        dataset = [{"input": "test input", "question": "test question"}]
        console = Console()

        # Mock metric
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "TestMetric"

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Reference field 'answer' not found in dataset"
        ):
            validate_output_mapping(
                dataset, "reference", "answer", [mock_metric], console
            )

    def test_validate_output_mapping_invalid_metric_parameter(self):
        """Test output mapping validation with invalid metric parameter."""
        # Mock dataset with valid fields
        dataset = [{"input": "test input", "answer": "test answer"}]
        console = Console()

        # Mock metric with invalid score method parameter
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "TestMetric"
        mock_metric.score = Mock()

        # Mock the signature to exclude 'reference' parameter
        from unittest.mock import Mock as MockSignature

        mock_sig = MockSignature()
        mock_sig.parameters = {"self": Mock(), "other_param": Mock()}

        with patch("inspect.signature", return_value=mock_sig):
            # Should not raise ValueError, just print warning
            validate_output_mapping(
                dataset, "reference", "answer", [mock_metric], console
            )

    def test_validate_output_mapping_empty_dataset(self):
        """Test output mapping validation with empty dataset."""

        # Mock empty dataset (iterator that yields nothing)
        class EmptyDataset:
            def __iter__(self):
                return iter([])

        dataset = EmptyDataset()
        console = Console()

        # Mock metric with proper score method signature
        class MockMetric:
            def score(self, reference=None):
                return 0.5

        mock_metric = MockMetric()

        # Should not raise an exception (returns early)
        validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)

    def test_validate_output_mapping_non_dict_items(self):
        """Test output mapping validation with non-dictionary items."""
        # Mock dataset with non-dict items
        dataset = ["not a dict", "also not a dict"]
        console = Console()

        # Mock metric with proper score method signature
        class MockMetric:
            def score(self, reference=None):
                return 0.5

        mock_metric = MockMetric()

        # Should not raise an exception (returns early)
        validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)

    def test_validate_output_mapping_metric_validation_error(self):
        """Test output mapping validation when metric parameter validation fails."""
        # Mock dataset with valid fields
        dataset = [{"input": "test input", "answer": "test answer"}]
        console = Console()

        # Mock metric that raises exception during parameter validation
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "TestMetric"

        # Mock inspect.signature to raise an exception
        with patch("inspect.signature", side_effect=Exception("Inspection failed")):
            # Should not raise an exception (continues with other metrics)
            validate_output_mapping(
                dataset, "reference", "answer", [mock_metric], console
            )


class TestOutputMappingParsing:
    """Test cases for output mapping parsing."""

    def test_parse_output_mapping_success(self):
        """Test successful output mapping parsing."""
        result = parse_output_mapping("reference=answer")
        assert result == ("reference", "answer")

    def test_parse_output_mapping_with_spaces(self):
        """Test output mapping parsing with spaces."""
        result = parse_output_mapping(" reference = answer ")
        assert result == ("reference", "answer")

    def test_parse_output_mapping_multiple_equals(self):
        """Test output mapping parsing with multiple equals signs."""
        result = parse_output_mapping("reference=field=with=equals")
        assert result == ("reference", "field=with=equals")

    def test_parse_output_mapping_no_equals(self):
        """Test output mapping parsing without equals sign."""
        with pytest.raises(
            ValueError, match="Output mapping must be in format reference=DATASET_FIELD"
        ):
            parse_output_mapping("reference")

    def test_parse_output_mapping_empty_string(self):
        """Test output mapping parsing with empty string."""
        with pytest.raises(
            ValueError, match="Output mapping must be in format reference=DATASET_FIELD"
        ):
            parse_output_mapping("")


class TestMetricsFallback:
    """Test cases for metrics fallback functionality."""

    def test_get_metrics_fallback_from_custom_to_opik(self):
        """Test that metrics fallback from custom file to opik.evaluation.metrics works."""
        # Create a temporary custom metrics file with only one metric
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
class CustomMetric:
    def score(self, reference=None, prediction=None):
        return 0.8
""")
            custom_metrics_file = f.name

        try:
            # Create mock config
            config = Mock()
            config.metric = "CustomMetric,Hallucination"  # CustomMetric exists in file, Hallucination should fallback
            config.metrics_file = custom_metrics_file
            config.debug = False
            config.config_path = "ez-config.json"  # Add proper config_path

            # Create evaluator instance
            evaluator = MCPEvaluator(config)

            # Mock the console to capture output
            evaluator.console = Mock()

            # Test the get_metrics method
            metrics = evaluator.get_metrics()

            # Should have 2 metrics
            assert len(metrics) == 2

            # Check that console was called with success messages for metrics
            metric_success_calls = [
                call
                for call in evaluator.console.print.call_args_list
                if "✅ Loaded metric:" in str(call)
            ]
            assert len(metric_success_calls) == 2

            # Verify one metric came from custom file and one from opik
            custom_call = [
                call
                for call in metric_success_calls
                if "custom metrics file" in str(call)
            ]
            opik_call = [
                call
                for call in metric_success_calls
                if "opik.evaluation.metrics" in str(call)
            ]

            assert len(custom_call) == 1
            assert len(opik_call) == 1

        finally:
            # Clean up temporary file
            os.unlink(custom_metrics_file)

    def test_get_metrics_fallback_error_when_not_found_anywhere(self):
        """Test that appropriate error is shown when metric is not found in either location."""
        # Create a temporary custom metrics file with only one metric
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
class CustomMetric:
    def score(self, reference=None, prediction=None):
        return 0.8
""")
            custom_metrics_file = f.name

        try:
            # Create mock config
            config = Mock()
            config.metric = "NonExistentMetric"  # This metric doesn't exist anywhere
            config.metrics_file = custom_metrics_file
            config.debug = False
            config.config_path = "ez-config.json"  # Add proper config_path

            # Create evaluator instance
            evaluator = MCPEvaluator(config)

            # Mock the console to capture output
            evaluator.console = Mock()

            # Test that ValueError is raised
            with pytest.raises(ValueError, match="Unknown metric: NonExistentMetric"):
                evaluator.get_metrics()

            # Check that error message was printed
            error_calls = [
                call
                for call in evaluator.console.print.call_args_list
                if "❌ Unknown metric" in str(call)
            ]
            assert len(error_calls) == 1

        finally:
            # Clean up temporary file
            os.unlink(custom_metrics_file)

    def test_get_metrics_no_custom_file_uses_opik_only(self):
        """Test that when no custom metrics file is provided, only opik metrics are used."""
        # Create mock config without metrics_file
        config = Mock()
        config.metric = (
            "Hallucination"  # This should be found in opik.evaluation.metrics
        )
        config.metrics_file = None
        config.debug = False
        config.config_path = "ez-config.json"  # Add proper config_path

        # Create evaluator instance
        evaluator = MCPEvaluator(config)

        # Mock the console to capture output
        evaluator.console = Mock()

        # Test the get_metrics method
        metrics = evaluator.get_metrics()

        # Should have 1 metric
        assert len(metrics) == 1

        # Check that console was called with success message from opik
        success_calls = [
            call
            for call in evaluator.console.print.call_args_list
            if "✅ Loaded metric" in str(call)
        ]
        assert len(success_calls) == 1

        # Verify metric came from opik
        opik_call = [
            call for call in success_calls if "opik.evaluation.metrics" in str(call)
        ]
        assert len(opik_call) == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
