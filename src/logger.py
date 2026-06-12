"""
This script sets up a standardized, automated logging system to record events, 
warnings, and errors while the program runs. It starts by generating a dynamically 
named log file based on the exact current date and time, ensuring previous logs 
are never accidentally overwritten. It then creates a dedicated 'logs' directory 
within the project folder to keep things organized. Finally, it configures Python's 
built-in `logging` module to write to this newly created file. The configuration 
includes a specific formatting rule so that every log entry automatically prints 
the exact timestamp, line number, logger name, severity level, and the custom message. 
This creates a detailed, permanent timeline of exactly what the program was doing, 
which is essential for debugging and monitoring health.
"""
# The 'logging' module is Python's built-in tool for keeping a diary of events.
import logging

# The 'os' module lets Python interact with your computer's operating system, 
# allowing it to do things like find folders or create new directories.
import os

# We import 'datetime' so we can access the computer's current clock and calendar.
from datetime import datetime


# We create a unique name for our text file using the exact current time.
# strftime('%m_%d_%Y_%M_%S') formats the time as: Month_Day_Year_Minute_Second.
# We add ".log" at the end to make it a standard log file.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%M_%S')}.log"

# We tell Python where we want to save this file. 
# os.getcwd() gets the "current working directory" (the folder you are in right now).
# We tell it to look in that folder, make a sub-folder called "logs", and use our filename.
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# This physically creates the folders on your computer if they don't exist yet.
# exist_ok=True means: "If this folder is already there, don't throw an error, just ignore this step."
os.makedirs(logs_path, exist_ok=True)

# Now we create the final, complete file path combining the folder path and the file name.
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# We configure the core settings for how our logging diary should behave globally.
logging.basicConfig(
    
    # We tell the logger to write everything into the specific file we just created.
    filename=LOG_FILE_PATH,
    
    # We define the template for what every single log entry should look like.
    # It will automatically stamp: [Time] LineNumber LoggerName - SeverityLevel - YourMessage
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    
    # We set the baseline severity level to INFO. This means it will record 
    # general information, warnings, and errors, but will ignore low-level "DEBUG" messages.
    level=logging.INFO,
)

# This line means: "Only run the test code below if I run this exact file directly."
# if __name__=="__main__":
    
    # We write our very first entry into the log file to prove it works!
#     logging.info("Logging has started")