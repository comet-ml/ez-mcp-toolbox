#!/usr/bin/env python3
"""
Tests for ez-mcp chatbot functionality.
These tests require a properly configured comet_ml.API() that can access real data.
"""

import pytest
from unittest.mock import patch, Mock
from io import StringIO

# Import the chatbot from the ez-mcp package
from ez_mcp_toolbox.chatbot import MCPChatbot


class TestChatbot:
    """Tests for the MCP chatbot functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # This would typically initialize the session for real API access
        # but for unit tests, we'll mock the dependencies
        pass

    @pytest.mark.asyncio
    async def test_chatbot_initialization(self):
        """Test chatbot can be initialized properly."""
        # Mock the config loading
        with patch("ez_mcp_toolbox.chatbot.MCPChatbot.load_config") as mock_load_config:
            mock_server = Mock(
                name="comet-mcp",
                description="Comet ML MCP server",
                command="comet-mcp",
                args=[],
                env=None,
            )
            mock_load_config.return_value = (
                [mock_server],
                "openai/gpt-4o-mini",
                {"temperature": 0.2},
            )

            # Mock the server connection
            with patch("ez_mcp_toolbox.chatbot.MCPChatbot.connect_all_servers"):
                # Mock the LLM completion to avoid real API calls
                with patch(
                    "ez_mcp_toolbox.utils.call_llm_with_tracing"
                ) as mock_completion:
                    # Mock the LLM response
                    mock_llm_response = Mock()
                    mock_choice = Mock()
                    mock_choice.message = Mock()
                    mock_choice.message.content = "The session is connected and ready."
                    mock_choice.message.tool_calls = None  # No tool calls for this test
                    mock_llm_response.choices = [mock_choice]
                    mock_completion.return_value = mock_llm_response

                    chatbot = MCPChatbot(
                        "ez-config.json",
                        system_prompt="You are a helpful AI assistant.",
                        max_rounds=2,
                    )

                    # Mock the session and tools
                    mock_session = Mock()
                    mock_tool = Mock()
                    mock_tool.name = "get_session_info"
                    mock_tool.description = "Get session information"
                    mock_tool.inputSchema = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }

                    mock_tools_response = Mock()
                    mock_tools_response.tools = [mock_tool]
                    mock_session.list_tools.return_value = mock_tools_response

                    mock_tool_result = Mock()
                    mock_tool_result.content = [
                        {
                            "type": "text",
                            "text": '{"initialized": true, "api_status": "Connected", "workspace": "test-user", "error": null}',
                        }
                    ]
                    mock_session.call_tool.return_value = mock_tool_result

                    chatbot.mcp_manager.sessions["comet-mcp"] = mock_session

                    # Test the chatbot functionality
                    response = await chatbot.chat("What is the current session status?")

                    # Verify response
                    assert isinstance(response, str)
                    assert len(response) > 0

                    print(f"Chatbot response: {response}")

    def test_rich_formatting(self):
        """Test that Rich formatting is properly initialized and working."""
        from ez_mcp_toolbox.chatbot import MCPChatbot
        from rich.console import Console

        # Create a chatbot instance
        chatbot = MCPChatbot(
            "ez-config.json", system_prompt="You are a helpful AI assistant."
        )

        # Verify that Rich console is initialized
        assert hasattr(chatbot, "console")
        assert isinstance(chatbot.console, Console)

        # Test that console can render markdown
        test_markdown = "# Test Header\n\nThis is **bold** text and *italic* text."

        # Capture console output
        console_output = StringIO()
        test_console = Console(file=console_output, force_terminal=True)
        test_console.print(test_markdown)

        # Verify output was captured (Rich formatting should work)
        output = console_output.getvalue()
        assert len(output) > 0

        print("✓ Rich formatting test passed")

    def test_opik_integration(self):
        """Test that Opik logging is properly integrated."""
        from ez_mcp_toolbox.chatbot import MCPChatbot
        import uuid

        # Create a chatbot instance
        chatbot = MCPChatbot(
            "ez-config.json", system_prompt="You are a helpful AI assistant."
        )

        # Verify that thread_id is generated
        assert hasattr(chatbot, "thread_id")
        assert isinstance(chatbot.thread_id, str)
        assert len(chatbot.thread_id) > 0

        # Verify that thread_id is a valid UUID
        try:
            uuid.UUID(chatbot.thread_id)
        except ValueError:
            assert False, "thread_id should be a valid UUID"

        # Verify that chat method has the @track decorator
        assert hasattr(
            chatbot.chat, "__wrapped__"
        ), "chat should be decorated with @track"

        print("✓ Opik integration test passed")
        print(f"  Thread ID: {chatbot.thread_id}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
