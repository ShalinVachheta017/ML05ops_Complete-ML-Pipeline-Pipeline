"""
Module: model_building
This module implements functionality for building a machine learning model.
It includes functions to load configuration parameters from a YAML file, load training data,
train a RandomForestClassifier model using the provided hyperparameters, and save the trained model to disk.
Logging is used throughout to help trace the execution and catch potential issues.
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# Ensure the "logs" directory exists for storing log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setup logging configuration for the model_building module
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

# Console handler outputs logs to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler stores logs to a file in the logs directory
log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter specifies the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding both handlers to the logger for comprehensive logging
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    This function reads the provided YAML file that contains configuration
    parameters for the model building process such as hyperparameters for the
    RandomForestClassifier. Logs the outcome of the operation.

    Parameters:
        params_path (str): The file path to the YAML configuration file.
    
    Returns:
        dict: A dictionary containing configuration parameters.
    
    Raises:
        FileNotFoundError: If the file at params_path does not exist.
        yaml.YAMLError: If there is an error while parsing the YAML file.
        Exception: For any other unexpected errors.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    This function reads a CSV file located at the provided file path and logs
    the shape of the loaded data. It is assumed that the CSV file contains the
    feature data for the model.

    Parameters:
        file_path (str): The file path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame containing the data.
    
    Raises:
        pd.errors.ParserError: If the CSV file is not formatted correctly.
        FileNotFoundError: If the CSV file is not found.
        Exception: For any other unforeseen errors.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model using the provided training data and hyperparameters.

    The function checks that the number of training samples matches between X_train and y_train,
    initializes the RandomForestClassifier with the given parameters, and fits the model to the data.
    Logs the start and completion of model training.

    Parameters:
        X_train (np.ndarray): A NumPy array of training features.
        y_train (np.ndarray): A NumPy array of training labels.
        params (dict): A dictionary containing model hyperparameters like 'n_estimators' and 'random_state'.
    
    Returns:
        RandomForestClassifier: The trained RandomForest model.
    
    Raises:
        ValueError: If the number of samples in X_train and y_train do not match.
        Exception: For any errors encountered during model initialization or training.
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state']) 
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file using pickle.

    This function ensures the destination directory exists before saving the model.
    The model is serialized using pickle and saved to the provided file path.

    Parameters:
        model: The trained model object to be saved.
        file_path (str): The file path where the model will be stored.
    
    Raises:
        FileNotFoundError: If the target directory of file_path is not found.
        Exception: For any errors during file writing or model serialization.
    """
    try:
        # Create the directory structure if it doesn't already exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    """
    Main function to execute the model building pipeline.

    This function performs the following steps:
      1. Loads model configuration parameters from a YAML file.
      2. Loads the processed training data.
      3. Extracts features and labels from the training DataFrame.
      4. Trains a RandomForestClassifier model using the training data.
      5. Saves the trained model to disk.
    
    Any exceptions are logged and printed to the console.
    """
    try:
        # Load configuration parameters specific to model building
        params = load_params('params.yaml')['model_building']
        
        # Load training data from CSV file containing TF-IDF processed features
        train_data = load_data('./data/processed/train_tfidf.csv')
        
        # Separate features and labels from the DataFrame
        X_train = train_data.iloc[:, :-1].values  # All columns except the last are features
        y_train = train_data.iloc[:, -1].values   # The last column is assumed to be the target label
        
        # Train the RandomForest model using the training data and parameters
        clf = train_model(X_train, y_train, params)
        
        # Define the path where the trained model will be saved
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
