import os
import time
from datetime import datetime, timedelta
import logging
import glob

def cleanup_old_logs(logs_dir: str, max_age_days: int = 30) -> None:
    """
    Delete log files older than the specified number of days.
    
    Args:
        logs_dir (str): Directory containing log files
        max_age_days (int): Maximum age of log files in days
    """
    if not os.path.exists(logs_dir):
        return

    # Calculate the cutoff time
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    # Get all log files
    log_patterns = [
        os.path.join(logs_dir, "app_*.log*"),  # Matches app logs and their rotated versions
        os.path.join(logs_dir, "performance_*.log*")  # Matches performance logs and their rotated versions
    ]
    
    deleted_files = 0
    
    for pattern in log_patterns:
        for log_file in glob.glob(pattern):
            try:
                # Get file's last modification time
                file_time = os.path.getmtime(log_file)
                
                if file_time < cutoff_time:
                    os.remove(log_file)
                    deleted_files += 1
                    
            except (OSError, PermissionError) as e:
                logging.getLogger('app').error(f"Error deleting old log file {log_file}: {e}")
    
    if deleted_files > 0:
        logging.getLogger('app').info(f"Cleaned up {deleted_files} log files older than {max_age_days} days")
