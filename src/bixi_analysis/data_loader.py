import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data_from_csv(filepath):
    """
    Load data from a CSV file located at the specified file path.
    
    Parameters:
    - filepath (str): The path to the CSV file to be loaded.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        print(f"The file at {filepath} was not found.")
        return None

def preprocess_data(data, columns_to_scale=None, date_columns=None):
    """
    Preprocess the data by converting date columns to datetime objects and
    scaling specified numeric columns using StandardScaler.
    
    Parameters:
    - data (DataFrame): The pandas DataFrame containing the data to be preprocessed.
    - columns_to_scale (list of str): List of numeric column names to be scaled. If None, no scaling is applied.
    - date_columns (list of str): List of date column names to be converted to datetime. If None, no conversion is applied.
    
    Returns:
    - DataFrame: The preprocessed pandas DataFrame.
    
    The function first checks if there are any date columns provided and then converts them to
    pandas datetime objects using `to_datetime`, which makes them compatible with ML models.
    For scaling, it utilizes StandardScaler from sklearn to standardize features by removing the
    mean and scaling to unit variance.
    """
    if date_columns:
        for col in date_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')  # Convert to datetime, invalid parsing will be set as NaT

    if columns_to_scale:
        scaler = StandardScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale].astype(float))  # Make sure data is in float format

    return data

# Now, using the function to load your data.
data = load_data_from_csv(r'C:/Users/PC/Downloads/OD_2019-07.csv')
