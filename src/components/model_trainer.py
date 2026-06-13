"""
This script handles the 'Model Training' phase of the machine learning pipeline.
It takes the newly transformed data arrays and splits them into input features (X) 
and the target variable (y). It then initializes a diverse dictionary of popular 
regression algorithms (like Random Forest, XGBoost, etc.) along with a grid of 
potential hyperparameters for each. By calling a custom evaluation utility, it tests 
every model and hyperparameter combination to find the absolute best performer 
based on the R-squared score. Finally, it saves the winning model as a physical 
'.pkl' file in the artifacts folder, ready to be deployed or used for predictions.
"""

import os
import sys
from dataclasses import dataclass

# Importing a variety of powerful machine learning algorithms for regression
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

# Importing our custom utility functions for saving files and evaluating models
from src.utils import save_object, evaluate_models


# As always, first we create a config class to manage paths and inputs.
# We use the @dataclass decorator so we don't need an explicit __init__ method.
@dataclass
class ModelTrainerConfig:
    # This specifies where the final, winning model will be saved.
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Automatically load the configuration path when the class is initialized.
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array): 
        """
        This function takes the fully processed numpy arrays (the output of data_transformer)
        and trains several models to find the best one.
        """
        try:
            logging.info("Splitting training and test input data")
            
            # Step 1: Split the arrays into X (inputs) and y (target)
            # train_array[:, :-1] means: take all rows, and all columns EXCEPT the last one (X_train)
            # train_array[:, -1] means: take all rows, and ONLY the last column (y_train)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            # Step 2: Create a dictionary containing all the models we want to test.
            # We instantiate them with their default parameters here.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # verbose=False stops CatBoost from printing hundreds of lines to the terminal during training
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Step 3: Define a dictionary of hyperparameters to test for EACH model.
            # Grid search will try every combination of these parameters to squeeze out the best performance.
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    # 'splitter': ['best','random'],
                    # 'max_features': ['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256] # Number of trees in the forest
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {}, # Linear regression doesn't have many hyperparameters to tune
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss': ['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            # Step 4: Evaluate all models and their parameters.
            # This custom function (evaluate_models) runs grid search cross-validation.
            # It returns a dictionary that looks like: {"Random Forest": 0.85, "Decision Tree": 0.72...}
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )    
            
            # Step 5: Find the highest R-squared score in the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Step 6: Find the exact NAME of the model that achieved that highest score
            # It finds the index of the best score in the values list, and pulls the corresponding key.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            # Step 7: Retrieve the actual, fully trained model object using that name
            best_model = models[best_model_name]

            # Step 8: Define a threshold. If our best model is worse than a 60% R2 score, 
            # we consider the whole training process a failure and raise an exception.
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            # Step 9: Save the winning model as a physical file so it can be deployed later.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Step 10: Run one final prediction on the test set using the best model
            predicted = best_model.predict(X_test)

            # Step 11: Calculate and return the final R-squared score
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            # Catch and log any errors

            raise CustomException(e, sys)

'''
How Hyperparameter Tuning is Happening Here
To understand this, we first need to separate parameters from hyperparameters:

Parameters are what the model learns on its own during training (e.g., the exact weight or mathematical multiplier assigned to the "reading_score" column).

Hyperparameters are the "settings" or "dials" you configure before the training even starts (e.g., telling a Random Forest exactly how many trees it is allowed to grow).

The code above automates the process of finding the perfect settings using a tool called GridSearchCV. Here is exactly how it works step-by-step:

1. Setting up the "Grid"
In your model_trainer.py file, you passed in a dictionary called param that contained lists of settings. For example, for a Random Forest, you passed:
'n_estimators': [8, 16, 32, 64, 128, 256]
This creates a "grid" of possible options. If you had multiple settings (like depth and learning rate), the grid would represent every single possible combination of those numbers.

2. The "Search" and "CV" (Cross-Validation)
When you run gs.fit(X_train, y_train), Scikit-Learn starts testing.
Because you set cv=3 (Cross-Validation = 3), it does something very clever:

It temporarily splits your X_train data into 3 smaller chunks.

It takes the first parameter option (e.g., 8 trees).

It trains the model on 2 of the chunks, and tests it on the 1 remaining chunk. It does this 3 times, rotating the chunks, and averages the score.

It then moves to the next parameter option (e.g., 16 trees) and repeats the entire process.

By doing this, it prevents the model from just memorizing the data, ensuring that the settings it chooses are actually good at generalizing.

3. Crowning the Winner
Once the search is over, GridSearchCV knows exactly which combination of settings yielded the highest average score. This winning combination is saved inside the variable gs.best_params_.

4. The Final Fit
Your code takes those winning settings and forcefully applies them to the base model using model.set_params(gs.best_params_). Finally, it trains the model one last time on the entire X_train dataset using those perfect settings (model.fit(X_train, y_train)).

Hyperparameter tuning can definitely feel like you are just throwing random numbers at a wall to see what sticks. Let's peel back the curtain and look at exactly what the numbers in our code are doing.

Think of a machine learning model as a highly complex soundboard in a recording studio. The **hyperparameters** are the physical dials and sliders. `GridSearchCV` is a robot producer that is systematically turning every dial, recording a track, scoring it, and eventually leaving the dials at the exact positions that produced the best-sounding song.

Here is exactly what the dials (numbers) in our specific `params` dictionary control for each model.

---

### 1. Decision Tree

A single decision tree works like a massive flowchart, asking yes/no questions to split your data until it reaches a prediction.

* **`criterion`: ['squared_error', 'absolute_error', 'poisson']**
* **What it is:** This isn't a number, but rather the mathematical rulebook the tree uses to decide *where* to split the data.
* **How it works:** `squared_error` tells the tree to heavily punish predictions that are way off the mark. `absolute_error` treats all mistakes equally, making it better if your data has weird outliers. `poisson` is specifically designed for predicting counts (like how many customers will visit a store). The grid search tests all three rulebooks to see which one the dataset responds to best.



### 2. Random Forest

A Random Forest is exactly what it sounds like: a literal forest made up of hundreds of individual Decision Trees. They all look at the data, make a prediction, and "vote" on the final answer.

* **`n_estimators`: [8, 16, 32, 64, 128, 256]**
* **What it is:** The exact number of trees in the forest.
* **How it works:** 8 means you have a tiny committee of 8 trees voting. 256 means a massive committee. Generally, more trees give you a more accurate and stable prediction, but it takes significantly more time and computer memory to train 256 trees compared to 8. Grid search finds the "sweet spot" where adding more trees stops improving the score.



### 3. The "Boosting" Models (Gradient Boosting, XGBoost, AdaBoost)

While a Random Forest builds hundreds of trees independently and averages them, **Boosting** builds trees sequentially. Tree #1 makes a prediction. It will inevitably make mistakes. Tree #2 is then built *specifically* to fix the mistakes of Tree #1. Tree #3 fixes Tree #2, and so on.

* **`n_estimators`: [8, 16, 32, 64, 128, 256]**
* **What it is:** The number of sequential trees built in the chain.


* **`learning_rate`: [0.1, 0.01, 0.05, 0.001]**
* **What it is:** How aggressive each new tree is allowed to be when correcting the previous tree's mistakes.
* **How it works:** If the learning rate is **0.1** (high), Tree #2 makes massive, sweeping corrections to Tree #1. If it is **0.001** (low), Tree #2 only makes microscopic, cautious tweaks. *Pro Tip: These two settings are tied together. If you use a tiny learning rate (0.001), you usually need a huge number of trees (256) to reach the final answer.*


* **`subsample`: [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]** *(Specific to Gradient Boosting in your code)*
* **What it is:** The percentage of your training data given to each tree.
* **How it works:** **0.6** means each individual tree is only allowed to look at a randomly selected 60% of your data. Forcing the trees to look at incomplete data actually makes the final model *smarter* because it prevents the model from just purely memorizing the dataset (a trap known as overfitting).



### 4. CatBoost Regressor

CatBoost is just another highly advanced "Boosting" algorithm (like XGBoost), but it uses slightly different terminology for its dials.

* **`iterations`: [30, 50, 100]**
* **What it is:** This is simply CatBoost's word for `n_estimators`. It dictates how many trees to build in the sequence.


* **`learning_rate`: [0.01, 0.05, 0.1]**
* **What it is:** Same as the other boosting models—how aggressively the trees correct each other.


* **`depth`: [6, 8, 10]**
* **What it is:** How many levels (or branches) deep each individual tree is allowed to grow.
* **How it works:** A depth of **6** means the tree can ask 6 consecutive "yes/no" questions before spitting out an answer. A depth of **10** allows for highly complex logic. However, if a tree goes too deep, it starts finding fake patterns that don't actually exist in the real world. Grid search tests these three depths to find the perfect balance between simple and complex.



---

### The Big Picture

When your script runs `GridSearchCV` on the **CatBoost** dictionary, for example, it is running a massive loop:

1. It tries 30 trees, at a 0.01 learning rate, with a depth of 6. Records the score.
2. It tries 30 trees, at a 0.05 learning rate, with a depth of 6. Records the score.
3. It tries 30 trees, at a 0.1 learning rate, with a depth of 6. Records the score...

It does this for *every single possible combination* of those numbers, for every single model, until it finds the absolute mathematical peak of your data's predictive power.
'''        