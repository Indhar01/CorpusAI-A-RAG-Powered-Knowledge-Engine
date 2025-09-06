import time
import functools
import logging
from typing import Callable, Any
import cProfile
import pstats
import io

perf_logger = logging.getLogger('performance')

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance using cProfile"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 time-consuming operations
        
        perf_logger.info(f"Performance profile for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

def measure_time(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        perf_logger.info(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

class PerformanceMonitor:
    """Context manager for monitoring code block performance"""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = end_time - self.start_time
        perf_logger.info(f"Operation '{self.operation_name}' took {execution_time:.4f} seconds")
