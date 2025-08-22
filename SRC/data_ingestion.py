import pandas as pd #dataframe operation
import os   #making directory
from sklearn.model_selection import train_test_split #for preprocessing and parting data points
import logging   
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
# Setting up a logger to track the execution of the script
logger = logging.getLogger('data_ingestion') # Logger for data ingestion module
logger.setLevel('DEBUG')  # Set the logging level to DEBUG to capture all messages from the script 

# Console handler for logging messages to the terminal
console_handler = logging.StreamHandler() # Console handler for logging to the terminal 
console_handler.setLevel('DEBUG') # Set the console logging level to DEBUG
 
# File handler for logging messages to a file
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatting the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # log format
console_handler.setFormatter(formatter)     # console 
file_handler.setFormatter(formatter)         # file 

# Adding handlers to the logger (where u need to log the details)
logger.addHandler(console_handler) # in the terminal
logger.addHandler(file_handler) # to a file

def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    
    Args:
        params_path (str): Path to the YAML file containing parameters.
    
    Returns:
        dict: Dictionary containing the parameters.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from  %s', params_path)
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

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        data_url (str): URL or path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by dropping unnecessary columns and renaming columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to preprocess.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    try:
        # Dropping unnecessary columns
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        # Renaming columns for better readability
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets to CSV files.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_path (str): Path to save the datasets.
    """
    try:
        # Create the directory for saving raw data if it doesn't exist
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        # Save train and test datasets as CSV files
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    """
    Main function to execute the data ingestion pipeline.
    """
    try:
        # Load parameters from the YAML file
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']  # Test size for splitting the data
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'  # URL of the dataset
        
        # Load the dataset
        df = load_data(data_url=data_path)
        
        # Preprocess the dataset
        final_df = preprocess_data(df)
        
        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test datasets
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()