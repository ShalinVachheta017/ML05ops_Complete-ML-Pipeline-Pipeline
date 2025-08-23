"""
Module: model_evaluation
This module handles the evaluation of a trained machine learning model.
It includes functions to load configuration parameters, load a trained model and test data,
compute evaluation metrics (accuracy, precision, recall, and AUC), and save the metrics to a JSON file.
Experiment tracking is integrated using dvclive.
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# ----------------------------------------------------------------------------
# Setup logging for the module to capture both console and file outputs.
# ----------------------------------------------------------------------------
# Create a directory for log files if it doesn't exist.
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Initialize the logger for model evaluation.
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Setup console handler for outputting logs to the terminal.
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Setup file handler for logging to a file.
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define the logging format.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach both console and file handlers to the logger.
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ----------------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------------

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.
    
    Parameters:
        params_path (str): The path to the YAML parameters file.
        
    Returns:
        dict: A dictionary containing the configuration parameters.
    
    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
        Exception: For any other unexpected errors.
    """
    try:
        # Open the YAML file and parse it into a Python dictionary.
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

def load_model(file_path: str):
    """
    Load a trained model from a pickle file.
    
    Parameters:
        file_path (str): The path to the pickle file containing the trained model.
        
    Returns:
        The deserialized model object.
    
    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: For any issues during model loading.
    """
    try:
        # Open the file in binary read mode and deserialize the model using pickle.
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load test data from a CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): The path to the CSV file containing the test data.
        
    Returns:
        pd.DataFrame: The loaded DataFrame with the test data.
    
    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        Exception: For any other errors during loading.
    """
    try:
        # Read the CSV file and convert it into a pandas DataFrame.
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the trained model and compute performance metrics.
    
    Metrics computed include accuracy, precision, recall, and AUC. The function uses the model
    to predict labels and corresponding probability estimates, and then calculates the metrics.
    
    Parameters:
        clf: The trained model to be evaluated.
        X_test (np.ndarray): The test set features.
        y_test (np.ndarray): The true labels of the test set.
        
    Returns:
        dict: A dictionary containing evaluation metrics.
    
    Raises:
        Exception: For any errors encountered during evaluation.
    """
    try:
        # Predict class labels for the test data.
        y_pred = clf.predict(X_test)
        # Predict the probabilities for the positive class.
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Compute evaluation metrics using sklearn.
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Store metrics in a dictionary.
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    
    Parameters:
        metrics (dict): The dictionary containing evaluation metrics.
        file_path (str): The destination file path where the metrics will be saved.
        
    Raises:
        Exception: For any issues during file writing.
    """
    try:
        # Ensure the target directory exists.
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write the metrics dictionary to a JSON file with indentation.
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """
    Main function to execute the model evaluation pipeline.
    
    Steps:
      1. Load evaluation parameters from the YAML configuration file.
      2. Load the trained model from disk.
      3. Load the test data for evaluation.
      4. Compute performance metrics using the loaded model and test data.
      5. Log experiment details using dvclive.
      6. Save the computed metrics to a JSON file.
    
    Exceptions are logged and printed if any step fails.
    """
    try:
        # Load configuration parameters.
        params = load_params(params_path='params.yaml')
        
        # Load the trained model.
        clf = load_model('./models/model.pkl')
        
        # Load the test dataset.
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        # Extract features and labels from the test data DataFrame.
        X_test = test_data.iloc[:, :-1].values  # All columns except the last are features.
        y_test = test_data.iloc[:, -1].values   # The last column is the target label.

        # Evaluate the model to compute performance metrics.
        metrics = evaluate_model(clf, X_test, y_test)

        # Experiment tracking using dvclive.
        # NOTE: Here, dummy metrics are logged for illustration.
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))
            live.log_params(params)
        
        # Save the computed metrics to a JSON file for reporting.
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()