"""
Tracing utilities for Azure Monitor OpenTelemetry integration.

This module provides a clean interface for tracing operations without cluttering
the business logic with telemetry details.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace


class TracingManager:
    """Manages Azure Monitor telemetry tracing."""

    def __init__(self):
        """Initialize the tracing manager."""
        self.tracer = None
        self._initialize()

    def _initialize(self):
        """Initialize Azure Monitor telemetry for tracing."""
        try:
            # Initialize AI Project client for telemetry
            project_client = AIProjectClient(
                credential=DefaultAzureCredential(),
                endpoint=os.environ["AZURE_AI_AGENT_PROJECT_ENDPOINT"],
            )

            # Get Application Insights connection string and configure monitoring
            connection_string = (
                project_client.telemetry.get_application_insights_connection_string()
            )
            configure_azure_monitor(connection_string=connection_string)

            # Initialize tracer
            self.tracer = trace.get_tracer(__name__)
            print("✅ Telemetry tracing initialized successfully")

        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize telemetry tracing: {e}")
            # Create a no-op tracer if telemetry fails
            self.tracer = trace.get_tracer(__name__)

    @contextmanager
    def trace_operation(
        self, operation_name: str, attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing an operation.

        Args:
            operation_name: Name of the operation to trace
            attributes: Optional dictionary of attributes to set on the span

        Yields:
            The span object for additional attribute setting if needed
        """
        with self.tracer.start_as_current_span(operation_name) as span:
            # Set initial attributes if provided
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
                # Mark as successful if no exception occurs
                if not span.is_recording():
                    return
                # Check if success attribute wasn't already set
                if attributes is None or "success" not in attributes:
                    span.set_attribute("success", True)
            except Exception as e:
                # Record the exception and mark as failed
                span.set_attribute("success", False)
                span.set_attribute("error.message", str(e))
                span.record_exception(e)
                raise

    def set_attributes(self, span, **kwargs):
        """
        Helper method to set multiple attributes on a span.

        Args:
            span: The span to set attributes on
            **kwargs: Key-value pairs to set as attributes
        """
        for key, value in kwargs.items():
            span.set_attribute(key, value)

    def trace_error(self, span, error_type: str, error_message: str):
        """
        Helper method to trace an error.

        Args:
            span: The span to record the error on
            error_type: Type/category of the error
            error_message: Error message
        """
        span.set_attribute("success", False)
        span.set_attribute("error.type", error_type)
        span.set_attribute("error.message", error_message)


# Singleton instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """
    Get or create the singleton TracingManager instance.

    Returns:
        The TracingManager instance
    """
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Convenience function to trace an operation using the global TracingManager.

    Args:
        operation_name: Name of the operation to trace
        attributes: Optional dictionary of attributes to set on the span

    Returns:
        Context manager for the traced operation
    """
    manager = get_tracing_manager()
    return manager.trace_operation(operation_name, attributes)
