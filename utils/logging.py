import os
import sys
import logging
from pathlib import Path

class DebugFormatter(logging.Formatter):
    """Custom formatter that ensures full message output"""
    
    def format(self, record):
        # Ensure the message is not truncated
        if hasattr(record, 'msg'):
            record.msg = str(record.msg)
            
            # If this is a debug message with a long content, format it properly
            if record.levelno == logging.DEBUG and len(record.msg) > 500:
                # Add proper line breaks and indentation for readability
                lines = record.msg.split('\n')
                record.msg = '\n    ' + '\n    '.join(lines)
        
        return super().format(record)

class LogConfig:
    """
    Logging configuration for FLARE Proxy.
    Handles log directory validation and logger setup.
    """
    
    def __init__(self, log_path: str, debug_mode: bool = False):
        """
        Initialize logging configuration.
        
        Args:
            log_path: Path to the log file
            debug_mode: Whether to enable debug logging to stdout
        """
        self.log_path = log_path
        self.debug_mode = debug_mode
        
    def validate_log_directory(self) -> None:
        """
        Validate that the log directory exists and is writable.
        Creates the directory if it doesn't exist.
        
        Raises:
            SystemExit: If the directory cannot be created or is not writable
        """
        log_dir = os.path.dirname(self.log_path)
        
        try:
            # Create directory if it doesn't exist
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            
            # Check if directory is writable
            if not os.access(log_dir, os.W_OK):
                print(f"Error: Log directory {log_dir} is not writable", file=sys.stderr)
                sys.exit(1)
                
        except Exception as e:
            print(f"Error creating log directory {log_dir}: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    def setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns:
            Logger instance configured for both file and optionally console output
        """
        # Validate log directory before setup
        self.validate_log_directory()
        
        # Create logger
        logger = logging.getLogger("flare_proxy")
        logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        debug_formatter = DebugFormatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for operational logging
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler for debug mode
        if self.debug_mode:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(debug_formatter)
            logger.addHandler(console_handler)
        
        return logger

class TrafficLogger:
    """
    Logger for capturing complete traffic between components.
    Only active in debug mode.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize traffic logger.
        
        Args:
            log_dir: Directory where the traffic.log will be created
        """
        self.logger = logging.getLogger("traffic")
        self.logger.setLevel(logging.INFO)
        
        # Create traffic.log file handler
        traffic_path = os.path.join(log_dir, 'traffic.log')
        traffic_handler = logging.FileHandler(traffic_path)
        
        # Use a simple formatter that won't truncate messages
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        traffic_handler.setFormatter(formatter)
        
        self.logger.addHandler(traffic_handler)
    
    def log_request(self, source: str, message: str):
        """Log a request with its source"""
        self.logger.info(f"REQUEST from {source}:\n{message}\n")
        
    def log_response(self, source: str, message: str):
        """Log a response with its source"""
        self.logger.info(f"RESPONSE from {source}:\n{message}\n")

def init_logging(log_path: str, debug_mode: bool = False) -> tuple[logging.Logger, TrafficLogger | None]:
    """
    Initialize logging for the application.
    
    Args:
        log_path: Path to the log file
        debug_mode: Whether to enable debug logging to stdout
        
    Returns:
        Configured logger instance
    """
    config = LogConfig(log_path, debug_mode)
    logger = config.setup_logging()
    
    # Initialize traffic logger if in debug mode
    traffic_logger = None
    if debug_mode:
        log_dir = os.path.dirname(log_path)
        traffic_logger = TrafficLogger(log_dir)
    
    return logger, traffic_logger
