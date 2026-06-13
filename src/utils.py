import os
import sys


import numpy as np
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


"""
This utility function evaluates a dictionary of machine learning models alongside a 
dictionary of their respective hyperparameters. It loops through each model, systematically 
tests different combinations of hyperparameters using Grid Search Cross-Validation, 
finds the absolute best settings for that specific model, trains it, and then evaluates 
its performance on the unseen test data. It returns a report (dictionary) containing 
the name of each model and its final R-squared score.
"""

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        # Create an empty dictionary that will eventually hold our results.
        # Example format: {"Random Forest": 0.85, "Decision Tree": 0.72}
        report = {}

        # Loop through the total number of models provided in the 'models' dictionary
        for i in range(len(list(models))):
            
            # 1. Grab the actual model object (e.g., RandomForestRegressor())
            model = list(models.values())[i]
            
            # 2. Grab the corresponding dictionary of hyperparameters for this specific model
            para = param[list(models.keys())[i]]

            # 3. Initialize GridSearchCV. 
            # We hand it the model, the parameters to test, and set cross-validation (cv) to 3.
            gs = GridSearchCV(model, para, cv=3)
            
            # 4. Run the grid search! This tests every combination of the parameters on the training data.
            gs.fit(X_train, y_train)

            # 5. gs.best_params_ contains the winning combination of parameters.
            # We inject these winning parameters back into our base model.
            model.set_params(**gs.best_params_)
            
            # 6. Train our base model one final time using these absolute best parameters.
            model.fit(X_train, y_train)

            # 7. Use the fully optimized, trained model to make predictions on the training data
            y_train_pred = model.predict(X_train)

            # 8. Use the fully optimized, trained model to make predictions on the unseen testing data
            y_test_pred = model.predict(X_test)

            # 9. Calculate the R-squared score for the training predictions
            train_model_score = r2_score(y_train, y_train_pred)

            # 10. Calculate the R-squared score for the testing predictions
            test_model_score = r2_score(y_test, y_test_pred)

            # 11. Save the test score into our report dictionary, using the model's name as the key.
            report[list(models.keys())[i]] = test_model_score

        # Once the loop has finished evaluating every model, return the final dictionary of scores.
        return report

    except Exception as e:
        # Catch any errors and trigger our custom exception handler
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)