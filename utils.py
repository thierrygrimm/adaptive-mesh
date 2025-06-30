import time
import os
import json
import logging

def current_time():
    """Get current timestamp in seconds.
    
    Returns:
        Current timestamp as a float
    """
    return time.time()

def get_repo_path(relative_path):
    """Get absolute path for a file relative to the repository root.
    
    Args:
        relative_path: Path relative to the repository root
        
    Returns:
        Absolute path to the file
    """
    # Get the directory of the current script (main.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)

def log_send(packet, log_file):
    """Log a sent packet to a JSON file.
    
    Args:
        packet: Dictionary containing packet data
        log_file: Path to the log file
    """
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Read existing logs or create new list
        existing_logs = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    existing_logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_logs = []
        
        # Add new packet
        existing_logs.append(packet)
        
        # Write back to file
        with open(log_file, 'w') as f:
            json.dump(existing_logs, f, indent=2)
            
    except Exception as e:
        logging.warning(f"Failed to log sent packet: {e}") 