"""
Module: model_evaluation
This module handles the evaluation of a trained machine learning model.
It includes functions to load parameters, load the trained model and test data,
evaluate the model (calculating metrics such as accuracy, precision, recall, and AUC),
and save these evaluation metrics to a JSON file.
It also integrates experiment tracking using dvclive.
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

# Ensure the "logs" directory exists for logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setup logging configuration for model evaluation
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Console handler for terminal output of logs
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler for writing logs to a file
log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter defines the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attaching handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.

    Reads the YAML file located at params_path containing parameters
    that can be used to configure evaluation settings like metric thresholds.
    Logs the retrieval process.

    Parameters:
        params_path (str): The path to the YAML parameters file.

    Returns:
        dict: A dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If an error occurs while parsing the YAML.
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

def load_model(file_path: str):
    """
    Load a trained model from a pickle file.

    This function deserializes the model stored in the provided file path.
    Ensures that any issues during deserialization are logged.

    Parameters:
        file_path (str): Path to the pickle file containing the trained model.

    Returns:
        The deserialized model object.

    Raises:
        FileNotFoundError: If the model file is not found.
        Exception: For any issues during loading.
    """
    try:
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

    This function reads the CSV file at the specified path,
    logs the successful data load, and returns the DataFrame.

    Parameters:
        file_path (str): The path to the CSV file containing test data.

    Returns:
        pd.DataFrame: DataFrame containing test data.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        Exception: For any other issues during data loading.
    """
    try:
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

    The function predicts labels and probabilities for the test dataset.
    It then calculates and logs metrics including accuracy, precision, recall, and AUC.

    Parameters:
        clf: The trained model to evaluate.
        X_test (np.ndarray): Test set features.
        y_test (np.ndarray): True labels for the test set.

    Returns:
        dict: Dictionary containing evaluation metrics.

    Raises:
        Exception: For any issues during model evaluation.
    """
    try:
        # Predict labels and calculate prediction probabilities for the positive class
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics for the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Organize metrics into a dictionary
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
    Save the evaluation metrics to a JSON file.

    This function ensures the target directory exists and writes the metrics
    dictionary to a JSON file with indentation for readability.

    Parameters:
        metrics (dict): Dictionary of evaluation metrics.
        file_path (str): The destination file path for the JSON file.

    Raises:
        Exception: For any issues during file writing.
    """
    try:
        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    """
    Execute the model evaluation pipeline.

    Main steps:
      1. Load configuration parameters from a YAML file.
      2. Load the trained model from disk.
      3. Load test data from a CSV file.
      4. Evaluate the model on the test data to compute performance metrics.
      5. Log experiment parameters and dummy metrics using dvclive for tracking.
      6. Save the evaluation metrics to a JSON file.

    Any exceptions encountered during the evaluation process are logged and printed.
    """
    try:
        # Load evaluation parameters
        params = load_params(params_path='params.yaml')
        
        # Load the trained model
        clf = load_model('./models/model.pkl')
        
        # Load the test dataset
        test_data = load_data('./data/processed/test_tfidf.csv')
        X_test = test_data.iloc[:, :-1].values  # Features: all columns except the last one
        y_test = test_data.iloc[:, -1].values   # Labels: the last column

        # Evaluate the model and calculate metrics
        metrics = evaluate_model(clf, X_test, y_test)

        # Experiment tracking using dvclive
        # NOTE: The following logging uses dummy metrics for precision, recall, etc.
        # You might want to replace y_test with appropriate predictions when tracking.
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))  # Dummy example
            live.log_metric('precision', precision_score(y_test, y_test))  # Dummy example
            live.log_metric('recall', recall_score(y_test, y_test))        # Dummy example
            live.log_params(params)
            logger.debug('Experiment tracking metrics logged via dvclive')
        
        # Save the computed evaluation metrics to a JSON report
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()