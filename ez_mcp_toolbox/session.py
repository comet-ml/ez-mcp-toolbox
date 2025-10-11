#!/usr/bin/env python3
"""
Session management for MCP server.
"""


class SessionContext:
    """Context manager for general session management."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the session context."""
        if self._initialized:
            return

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the session is initialized."""
        return self._initialized

    def reset(self) -> None:
        """Reset the session (useful for testing or reconfiguration)."""
        self._initialized = False


# Global session context instance
session_context = SessionContext()


def initialize_session() -> None:
    """Initialize the global session context."""
    session_context.initialize()


def get_session_context() -> SessionContext:
    """Get the global session context instance."""
    return session_context
