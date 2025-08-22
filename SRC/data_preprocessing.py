"""
Module: data_preprocessing
This module contains functions for loading raw data, preprocessing text data, encoding target labels,
and saving the processed data. It uses logging to record the processing steps and potential errors, 
making it easier to debug the preprocessing pipeline.

Functions:
    transform_text(text): Normalize and transform input text.
    preprocess_df(df, text_column, target_column): Preprocess the DataFrame by encoding and cleaning.
    main(text_column, target_column): Main procedure which reads, cleans, and saves the data.
"""

import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK resources if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Ensure the "logs" directory exists to store log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Setting up logger for better traceability
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Console handler outputs logs to the terminal
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler stores logs to a file for persistent logging
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter defines the structure of the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text string by performing the following operations:
    
    1. Converts the text to lowercase.
    2. Tokenizes the text into individual words.
    3. Removes non-alphanumeric tokens.
    4. Eliminates stopwords and punctuation.
    5. Applies stemming to each remaining word using the PorterStemmer.
    6. Joins the processed tokens back into a single space-separated string.
    
    Parameters:
        text (str): The input text string to be transformed.
        
    Returns:
        str: The processed text string.
    """
    # Initialize the Porter Stemmer for stemming words
    ps = PorterStemmer()
    
    # Convert the text to lowercase to standardize it
    text = text.lower()
    
    # Tokenize the text into a list of words using nltk's tokenizer
    text = nltk.word_tokenize(text)
    
    # Filter tokens: keep only alphanumeric words 
    text = [word for word in text if word.isalnum()]  # isalnum() checks if a string contains only alphanumeric characters 
    # ALPHANUMERIC MEANS LETTERS AND NUMBERS
    
    # Remove stopwords and punctuation from the tokens
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Apply stemming to each token to reduce words to their base form
    text = [ps.stem(word) for word in text]
    
    # Combine the tokens back into a single string and return it
    return " ".join(text)

def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the input DataFrame by encoding the target column, removing duplicates,
    and transforming the text column using the transform_text function.
    
    The function performs the following steps:
    1. Encodes the target column using a LabelEncoder.
    2. Removes duplicate rows in the DataFrame.
    3. Applies text transformation on the specified text column.
    
    Parameters:
        df (pd.DataFrame): The raw data DataFrame.
        text_column (str): Name of the column containing text to be processed.
        target_column (str): Name of the column representing the target labels.
        
    Returns:
        pd.DataFrame: The DataFrame after preprocessing.
    
    Raises:
        KeyError: If specified text or target column is not found in the DataFrame.
        Exception: For any other errors during processing.
    """
    try:
        logger.debug('Starting preprocessing for the provided DataFrame')
        
        # Encode the target column (convert categorical labels to numeric)
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded successfully')

        # Remove duplicate rows to ensure data quality
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicate rows removed')
        
        # Apply text transformation on the text column using the transform_text function
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed successfully')
        
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

def main(text_column='text', target_column='target'):
    """
    Main function for the data preprocessing pipeline.
    
    It performs the following tasks:
    1. Reads the training and testing datasets from the data/raw directory.
    2. Preprocesses both datasets (encoding target labels, removing duplicates, and transforming text).
    3. Saves the processed data as CSV files in the data/interim directory.
    
    Parameters:
        text_column (str): The name of the column containing text data. Defaults to 'text'.
        target_column (str): The name of the column containing target labels. Defaults to 'target'.

    Exception Handling:
        Logs an error and stops execution if files are not found, if the data is empty, or any other error occurs.
    """
    try:
        # Read raw training and testing data from CSV files
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded from CSV files')

        # Preprocess the training and testing DataFrames
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Define the directory where the processed data will be stored
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        # Save the processed DataFrames to CSV files
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug('Processed data saved successfully to %s', data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data found in the file: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

# Entry-point for the script
if __name__ == '__main__':
    main()