"""External assembly observation and a single-canvas workflow interface."""
from .adapter import assemble_with_trace
from .recorder import TraceReader, TraceRecorder
from .server import open_workbench, start_ui, start_server

__all__ = ["assemble_with_trace", "open_workbench", "start_ui", "start_server", "TraceReader", "TraceRecorder"]
