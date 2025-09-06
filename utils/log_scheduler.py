import schedule
import time
import threading
from .log_cleanup import cleanup_old_logs
import os

def start_cleanup_scheduler():
    """
    Start a background thread that runs the cleanup job daily
    """
    def cleanup_job():
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        cleanup_old_logs(logs_dir)
    
    def run_scheduler():
        # Schedule cleanup to run daily at midnight
        schedule.every().day.at("00:00").do(cleanup_job)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
    
    # Start the scheduler in a background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
