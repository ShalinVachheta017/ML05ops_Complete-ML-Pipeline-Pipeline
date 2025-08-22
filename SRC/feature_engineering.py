"""
Module: feature_engineering
This module performs feature engineering on text data by applying a TF-IDF transformation.
It involves loading parameters from a YAML file, reading preprocessed CSV data,
applying the TF-IDF vectorization on textual data, and saving the transformed arrays back to CSV files.
Detailed logging is provided for each step to facilitate debugging and traceability.

Functions:
    load_params(params_path): Loads configuration parameters from a YAML file.
    load_data(file_path): Loads a dataset from a CSV file, filling missing values with empty strings.
    apply_tfidf(train_data, test_data, max_features): Applies TF-IDF vectorization to training and testing data.
    save_data(df, file_path): Saves the provided DataFrame to a CSV file.
    main(): Orchestrates the feature engineering pipeline.
"""

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists for storing log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setup logging configuration for the feature_engineering module
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# Console handler for real-time log output in the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler for persistent logging to a log file
log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter to specify the log message format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Load configuration parameters from a YAML file.
    
    Reads from the given YAML file path and returns a dictionary containing the parameters.
    Logging is performed to indicate successful retrieval or any issues encountered.

    Parameters:
        params_path (str): The file path to the YAML parameters file.
        
    Returns:
        dict: A dictionary containing the configuration parameters.
        
    Raises:
        FileNotFoundError: If the YAML file is not found.
        yaml.YAMLError: If an error occurs while parsing the YAML file.
        Exception: For any other unexpected errors.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error encountered: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading parameters: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Reads a CSV file located at the specified file path and fills any missing values with empty strings.
    This helps prevent issues with null values during further text processing.

    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame with missing values replaced by empty strings.
        
    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed correctly.
        Exception: For any other unforeseen errors.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s and missing values filled', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file at %s: %s', file_path, e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading data from %s: %s', file_path, e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TF-IDF transformation to text data in the training and testing datasets.
    
    Uses sklearn's TfidfVectorizer to convert textual data into numerical features based on term frequency-inverse document frequency.
    This function extracts the 'text' and 'target' columns from both datasets, vectorizes the text, and then combines the results
    with their respective labels into new DataFrames.

    Parameters:
        train_data (pd.DataFrame): Training dataset with columns 'text' and 'target'.
        test_data (pd.DataFrame): Testing dataset with columns 'text' and 'target'.
        max_features (int): The maximum number of features (vocabulary words) to consider during vectorization.
        
    Returns:
        tuple: A tuple containing two DataFrames (train_df, test_df) with TF-IDF features and label columns.
        
    Raises:
        Exception: If any error occurs during the TF-IDF transformation process.
    """
    try:
        # Initialize the TF-IDF Vectorizer with specified maximum features
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract text and labels from training and testing datasets
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        # Fit the vectorizer on training data and transform both training and testing text data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Convert the sparse matrices to DataFrames and append the label columns
        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.debug('TF-IDF transformation applied successfully')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Ensures that the directory structure for the file path exists before writing the DataFrame to a CSV file.
    The file is saved without the index to maintain clean CSV formatting.

    Parameters:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The destination path for the CSV file.
        
    Raises:
        Exception: If any error occurs during saving.
    """
    try:
        # Create folder structure if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved successfully to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving data to %s: %s', file_path, e)
        raise

def main():
    """
    Execute the feature engineering pipeline.
    
    Steps:
      1. Load configuration parameters from a YAML file.
      2. Load preprocessed training and testing data.
      3. Apply TF-IDF transformation to the text data.
      4. Save the resulting feature data to CSV files.
      
    Any errors during the process are logged and the error details are printed to the console.
    """
    try:
        # Load parameters from the YAML configuration file
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        # You can uncomment the following line to manually set max_features if needed:
        # max_features = 50

        # Load preprocessed training and testing datasets
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Apply TF-IDF vectorization to convert text data into numerical features
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save the transformed data as CSV files in the "processed" directory
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()