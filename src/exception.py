# The 'sys' module gives us access to system-level information, 
# specifically the details about errors that happen in the background.
import sys

# The 'logging' module allows us to keep a running record (a log) 
# of events or errors that happen while the program is running.
from src.logger import logging


# We create a helper function to build our detailed error message.
# It needs the error itself, and the 'sys' tool to investigate it.
def error_message_detail(error, error_detail:sys):
    
    # sys.exc_info() returns 3 pieces of info about the current error.
    # We ignore the first two using underscores (_), and save the 3rd 
    # piece (the traceback), which contains the line number and file info.
    _, _, exc_tb = error_detail.exc_info()
    
    # We dig inside the traceback object to extract the exact name of the file
    # where the code failed.
    file_name = exc_tb.tb_frame.f_code.co_filename 
    
    # We create a custom text message. We use .format() to fill in the blanks 
    # with the file name, the line number (exc_tb.tb_lineno), and the actual error.
    error_message = "error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    # We send this fully detailed text message back so it can be used.
    return error_message


# We create our own custom error blueprint. By putting (Exception) in the 
# parentheses, we are telling Python: "Inherit all the features of a normal error."
class CustomException(Exception):
    
    # This is the setup method. It runs automatically when this custom error is triggered.
    def __init__(self, error_message, error_detail:sys):
        
        # This tells the base 'Exception' class to do its normal setup first.
        super().__init__(error_message)
        
        # We call our helper function from above to build the highly detailed 
        # text message, and we save it inside this class instance.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    # This method controls what gets printed if someone tries to print this error to the screen.
    def __str__(self):
        
        # Instead of printing a default Python error, we tell it to print our custom detailed message.
        return self.error_message    


# This line simply means: "Only run the code below if I am running this specific file directly."
# if __name__=="__main__":

    # The 'try' block tells Python: "Attempt to run this code, but be prepared if it fails."
#     try:
        # We purposely try to divide by zero to force a math error.
#         a=1/0
        
    # If the code above fails, Python jumps down here instead of crashing the whole program.
    # We catch the error and temporarily name it 'e'.
#     except Exception as e:
        
        # We write down a note in our log file that this happened.
#         logging.info("Divide by zero error")
        
        # Finally, we trigger (raise) our custom error. We pass it the original error 'e' 
        # and the 'sys' tool so it can find the file name and line number!
#         raise CustomException(e, sys)

if __name__=="__main__":

    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by zero error")
        raise CustomException (e,sys)   