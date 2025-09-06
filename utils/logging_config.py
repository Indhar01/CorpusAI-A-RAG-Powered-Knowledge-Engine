import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from .log_cleanup import cleanup_old_logs
import atexit

class MonthlyRotatingFileHandler(RotatingFileHandler):
    """Custom file handler that includes the month in the filename"""
    def __init__(self, filename_base, *args, **kwargs):
        self.filename_base = filename_base
        current_month = datetime.now().strftime('%Y%m')
        filename = f"{filename_base}_{current_month}.log"
        super().__init__(filename, *args, **kwargs)
    
    def doRollover(self):
        """
        Do a rollover checking if month has changed
        """
        current_month = datetime.now().strftime('%Y%m')
        new_filename = f"{self.filename_base}_{current_month}.log"
        
        if self.baseFilename != new_filename:
            # Month has changed, update filename
            if self.stream:
                self.stream.close()
                self.stream = None
            
            self.baseFilename = new_filename
            self.mode = 'a'
            self.stream = self._open()
        
        super().doRollover()

def setup_logging():
    """
    Configure application-wide logging with separate app and performance logs,
    including automatic cleanup of old logs.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Clean up old logs on startup
    cleanup_old_logs(logs_dir)
    
    # Register cleanup on application exit
    atexit.register(lambda: cleanup_old_logs(logs_dir))
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up app logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    
    # Set up app file handler with monthly rotation
    app_file_handler = MonthlyRotatingFileHandler(
        filename_base=os.path.join(logs_dir, 'app'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    app_file_handler.setFormatter(log_format)
    app_logger.addHandler(app_file_handler)
    
    # Set up app console handler
    app_console_handler = logging.StreamHandler()
    app_console_handler.setFormatter(log_format)
    app_logger.addHandler(app_console_handler)
    
    # Set up performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False
    
    # Set up performance file handler with monthly rotation
    perf_file_handler = MonthlyRotatingFileHandler(
        filename_base=os.path.join(logs_dir, 'performance'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    perf_file_handler.setFormatter(log_format)
    perf_logger.addHandler(perf_file_handler)
    
    # Set up performance console handler
    perf_console_handler = logging.StreamHandler()
    perf_console_handler.setFormatter(log_format)
    perf_logger.addHandler(perf_console_handler)
    
    return app_logger, perf_logger