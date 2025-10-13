#!/usr/bin/env python3
"""
Unit tests for ez_mcp_toolbox.evaluator module.
Tests the validation functionality for input and output fields.
"""

import pytest
from unittest.mock import Mock, patch
from rich.console import Console

# Import the evaluator module
from ez_mcp_toolbox.evaluator import (
    validate_input_field,
    validate_output_mapping,
    parse_output_mapping
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
        with pytest.raises(ValueError, match="Input field 'input' not found in dataset"):
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
        import inspect
        from unittest.mock import Mock as MockSignature
        mock_sig = MockSignature()
        mock_sig.parameters = {
            'self': Mock(),
            'reference': Mock(),
            'other_param': Mock()
        }
        
        with patch('inspect.signature', return_value=mock_sig):
            # Should not raise an exception
            validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)

    def test_validate_output_mapping_missing_reference_field(self):
        """Test output mapping validation with missing reference field."""
        # Mock dataset without the reference field
        dataset = [{"input": "test input", "question": "test question"}]
        console = Console()
        
        # Mock metric
        mock_metric = Mock()
        mock_metric.__class__.__name__ = "TestMetric"
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Reference field 'answer' not found in dataset"):
            validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)

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
        import inspect
        from unittest.mock import Mock as MockSignature
        mock_sig = MockSignature()
        mock_sig.parameters = {
            'self': Mock(),
            'other_param': Mock()
        }
        
        with patch('inspect.signature', return_value=mock_sig):
            # Should not raise ValueError, just print warning
            validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)

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
        with patch('inspect.signature', side_effect=Exception("Inspection failed")):
            # Should not raise an exception (continues with other metrics)
            validate_output_mapping(dataset, "reference", "answer", [mock_metric], console)


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
        with pytest.raises(ValueError, match="Output mapping must be in format reference=DATASET_FIELD"):
            parse_output_mapping("reference")

    def test_parse_output_mapping_empty_string(self):
        """Test output mapping parsing with empty string."""
        with pytest.raises(ValueError, match="Output mapping must be in format reference=DATASET_FIELD"):
            parse_output_mapping("")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
