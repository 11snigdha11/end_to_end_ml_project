"""
This script handles the 'Data Ingestion' phase of the machine learning pipeline. 
Its primary responsibility is to read the raw dataset from a specific source 
(like a local file, or a database), and intelligently split that data into 
training and testing sets. To maintain a clean and reproducible workflow, it 
creates an 'artifacts' directory where it saves copies of the raw data, the 
training data, and the testing data. By saving these physical files, downstream 
components (like data transformation and model training) can easily locate and 
use the exact same datasets without having to re-read or re-split the original source.
"""

# Import necessary core python modules for system and file path operations
import os
import sys

# Import our custom exception handler and logger to track errors and events
from src.exception import CustomException
from src.logger import logging

# Import Pandas for handling tabular data (dataframes)
import pandas as pd

# Import scikit-learn's tool for easily splitting data into train and test sets
from sklearn.model_selection import train_test_split

# Import dataclass, a special decorator that helps us define classes 
# specifically for storing data/variables without needing to write an __init__ method.
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# The @dataclass decorator automatically generates the __init__ method for us behind the scenes.
@dataclass  
class DataIngestionConfig:
    """
    This configuration class stores all the file paths that the DataIngestion 
    class will need. By default, it sets up paths inside an 'artifacts' folder.
    """
    # Path where the training data will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv") 
    
    # Path where the testing data will be saved
    test_data_path: str = os.path.join('artifacts', "test.csv")
    
    # Path where a complete copy of the raw data will be saved
    raw_data_path: str = os.path.join('artifacts', "stud.csv")


class DataIngestion:
    """
    This is the main class responsible for the actual data ingestion process.
    """
    def __init__(self):
        # When this class is created, it automatically loads the file paths defined above.
        # self.ingestion_config will now hold train_data_path, test_data_path, and raw_data_path.
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        This method executes the data ingestion pipeline: reading data, saving raw data, 
        splitting into train/test, and saving the splits.
        """
        # Log that the process has officially started
        logging.info("Entered the data ingestion method or component")

        try:
            # READ THE DATA
            # Read the initial dataset into a pandas DataFrame. 
            # (Note: In an industry setting, this line could be replaced with code to read from MongoDB, MySQL, etc.)
            df = pd.read_csv('notebook/data/stud.csv') 
            
            # Log that reading was successful
            logging.info('Read the dataset as dataframe')

            # CREATE DIRECTORIES
            # Get the directory name from the train_data_path (which is 'artifacts').
            # os.makedirs creates that folder. exist_ok=True means it won't crash if the folder already exists.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # SAVE RAW DATA
            # Save a copy of the complete dataframe into the artifacts folder as 'stud.csv'
            # index=False ensures we don't save pandas' row numbers as a column.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # SPLIT THE DATA
            logging.info("Train test split initiated")
            # Split the dataframe: 80% for training, 20% for testing. 
            # random_state=42 ensures the split is the exact same every time we run the code.
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # SAVE SPLIT DATA
            # Save the 80% training data to the 'train.csv' path in artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save the 20% testing data to the 'test.csv' path in artifacts
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)   
            
            logging.info("Ingestion of data is completed")

            # RETURN THE PATHS
            # We return the paths to the newly created train and test files.
            # The next component in the pipeline (Data Transformation) will use these paths to grab the data.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path    
            )
             
        except Exception as e:
            # If anything fails in the try block, our custom exception captures the error 
            # and the sys module pinpoints exactly where it happened.
            raise CustomException(e, sys)

# This testing block runs only if you run this specific file directly (e.g., `python src/components/data_ingestion.py`)
if __name__ == "__main__":
    
    # Create an instance of the DataIngestion class
    obj = DataIngestion()
    
    # Trigger the ingestion process
    train_data,test_data=obj.initiate_data_ingestion()


    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))