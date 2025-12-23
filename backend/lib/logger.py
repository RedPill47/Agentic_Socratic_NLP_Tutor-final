"""
Enhanced Logging Utility for Backend

Provides user-friendly, structured logging with:
- Color-coded log levels
- Pretty printing for complex data
- Clear section separators
- Timing information
- Easy error detection
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pprint import pformat

# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Log levels
    DEBUG = '\033[36m'      # Cyan
    INFO = '\033[32m'       # Green
    WARNING = '\033[33m'    # Yellow
    ERROR = '\033[31m'      # Red
    CRITICAL = '\033[35m'   # Magenta
    
    # Sections
    SECTION = '\033[94m'    # Bright Blue
    SUBSECTION = '\033[96m' # Bright Cyan
    
    # Data
    KEY = '\033[93m'        # Bright Yellow
    VALUE = '\033[92m'      # Bright Green
    TIMESTAMP = '\033[90m'  # Dark Gray


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and enhanced formatting."""
    
    # Icons for different log types
    ICONS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
    }
    
    # Section icons
    SECTION_ICONS = {
        'REQUEST': '📥',
        'RESPONSE': '📤',
        'SESSION': '💾',
        'RAG': '📚',
        'MAS': '🤖',
        'ONBOARDING': '🎓',
        'PLANNING': '📋',
        'ERROR': '❌',
        'SUCCESS': '✅',
        'INFO': 'ℹ️',
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and icons."""
        # Get icon based on section or level
        icon = self.SECTION_ICONS.get(record.name.split('.')[-1], self.ICONS.get(record.levelname, '•'))
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Get color for log level
        if self.use_colors:
            level_color = {
                'DEBUG': Colors.DEBUG,
                'INFO': Colors.INFO,
                'WARNING': Colors.WARNING,
                'ERROR': Colors.ERROR,
                'CRITICAL': Colors.CRITICAL,
            }.get(record.levelname, Colors.RESET)
            reset = Colors.RESET
            timestamp_color = Colors.TIMESTAMP
        else:
            level_color = reset = timestamp_color = ''
        
        # Format level name
        level_name = f"{level_color}{record.levelname:8s}{reset}"
        
        # Format message
        message = record.getMessage()
        
        # Check if message contains structured data (JSON-like)
        if '{' in message and '}' in message:
            try:
                # Try to pretty print if it looks like JSON
                if message.strip().startswith('{') or message.strip().startswith('['):
                    data = json.loads(message)
                    message = f"\n{pformat(data, indent=2, width=100)}"
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Build formatted log line
        formatted = (
            f"{timestamp_color}[{timestamp}]{reset} "
            f"{icon} {level_name} "
            f"{Colors.BOLD if self.use_colors else ''}{record.name}{Colors.RESET if self.use_colors else ''} "
            f"| {message}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class StructuredLogger:
    """Structured logger with section grouping and pretty printing."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger(name)
        self._section_stack = []
    
    def _format_data(self, data: Any, indent: int = 2) -> str:
        """Format data structure for pretty printing."""
        if isinstance(data, dict):
            formatted_items = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    formatted_value = self._format_data(value, indent + 2)
                    formatted_items.append(f"{' ' * indent}{Colors.KEY if Colors.KEY else ''}{key}{Colors.RESET if Colors.RESET else ''}: {formatted_value}")
                else:
                    formatted_items.append(f"{' ' * indent}{Colors.KEY if Colors.KEY else ''}{key}{Colors.RESET if Colors.RESET else ''}: {Colors.VALUE if Colors.VALUE else ''}{value}{Colors.RESET if Colors.RESET else ''}")
            return "{\n" + "\n".join(formatted_items) + f"\n{' ' * (indent - 2)}}}"
        elif isinstance(data, list):
            if len(data) > 5:
                # Truncate long lists
                items = [self._format_data(item, indent + 2) for item in data[:3]]
                newline = '\n'
                indent_str = ' ' * (indent - 2)
                items_str = f',{newline}'.join(items)
                return f"[{newline}{items_str}{newline}{indent_str}... ({len(data)} items total){newline}{indent_str}]"
            else:
                items = [self._format_data(item, indent + 2) for item in data]
                newline = '\n'
                indent_str = ' ' * (indent - 2)
                items_str = f',{newline}'.join(items)
                return f"[{newline}{items_str}{newline}{indent_str}]"
        else:
            return str(data)
    
    def section(self, title: str, data: Optional[Dict[str, Any]] = None):
        """Start a new log section."""
        self._section_stack.append(title)
        separator = "=" * 80
        if sys.stdout.isatty():
            print(f"\n{Colors.SECTION}{separator}{Colors.RESET}")
            print(f"{Colors.SECTION}📋 {title.upper()}{Colors.RESET}")
            if data:
                print(f"{Colors.SUBSECTION}{self._format_data(data)}{Colors.RESET}")
            print(f"{Colors.SECTION}{separator}{Colors.RESET}\n")
        else:
            print(f"\n{separator}")
            print(f"📋 {title.upper()}")
            if data:
                print(self._format_data(data))
            print(f"{separator}\n")
    
    def subsection(self, title: str, data: Optional[Dict[str, Any]] = None):
        """Start a subsection within current section."""
        separator = "-" * 60
        if sys.stdout.isatty():
            print(f"\n{Colors.SUBSECTION}{separator}{Colors.RESET}")
            print(f"{Colors.SUBSECTION}  → {title}{Colors.RESET}")
            if data:
                print(f"{Colors.SUBSECTION}{self._format_data(data)}{Colors.RESET}")
            print(f"{Colors.SUBSECTION}{separator}{Colors.RESET}\n")
        else:
            print(f"\n{separator}")
            print(f"  → {title}")
            if data:
                print(self._format_data(data))
            print(f"{separator}\n")
    
    def end_section(self):
        """End current section."""
        if self._section_stack:
            self._section_stack.pop()
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug message with optional data."""
        if data:
            self.logger.debug(f"{message}\n{self._format_data(data)}")
        else:
            self.logger.debug(message)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message with optional data."""
        if data:
            self.logger.info(f"{message}\n{self._format_data(data)}")
        else:
            self.logger.info(message)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message with optional data."""
        if data:
            self.logger.warning(f"{message}\n{self._format_data(data)}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, error: Optional[Exception] = None, data: Optional[Dict[str, Any]] = None):
        """Log error message with exception and optional data."""
        error_info = f"Error: {type(error).__name__}: {str(error)}" if error else ""
        if data:
            self.logger.error(f"{message} {error_info}\n{self._format_data(data)}", exc_info=error)
        else:
            self.logger.error(f"{message} {error_info}", exc_info=error)
    
    def success(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log success message."""
        if data:
            self.logger.info(f"✅ {message}\n{self._format_data(data)}")
        else:
            self.logger.info(f"✅ {message}")
    
    def request(self, method: str, path: str, user_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """Log incoming request."""
        request_data = {
            "method": method,
            "path": path,
            "user_id": user_id[:20] + "..." if user_id and len(user_id) > 20 else user_id,
        }
        if data:
            request_data.update(data)
        self.logger.info(f"📥 REQUEST: {method} {path}", extra={"data": request_data})
    
    def response(self, status: int, path: str, duration: Optional[float] = None, data: Optional[Dict[str, Any]] = None):
        """Log response."""
        response_data = {
            "status": status,
            "path": path,
            "duration_ms": f"{duration * 1000:.2f}" if duration else None,
        }
        if data:
            response_data.update(data)
        self.logger.info(f"📤 RESPONSE: {status} {path}", extra={"data": response_data})


def setup_logging(level: int = logging.INFO, use_colors: bool = True):
    """Setup enhanced logging configuration."""
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(use_colors=use_colors))
    
    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy loggers
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name, logging.getLogger(name))

