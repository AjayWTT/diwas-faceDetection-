import logging
import os
from datetime import datetime

# Set the log file name with current date and time
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"

# Set the log file directory
LOG_FILE_DIR = os.path.abspath(os.path.join(os.getcwd(), "logs"))

# Check if the log file directory exists, and create it if it doesn't
if not os.path.isdir(LOG_FILE_DIR):
    os.mkdir(LOG_FILE_DIR)

# Construct the log file path
LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

# Initialize the custom handler for logging
handler = logging.FileHandler(LOG_FILE_PATH, mode='w')

# Format the log entries
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Get the root logger
root_logger = logging.getLogger()

# Add the custom handler to the root logger
root_logger.addHandler(handler)

# Configure the logging level
root_logger.setLevel(logging.INFO)
