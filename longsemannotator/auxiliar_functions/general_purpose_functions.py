import pickle
import os

def load_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): The path to the pickle file to be loaded.

    Returns:
    object: The data loaded from the pickle file, or None if an error occurs.

    Raises:
    FileNotFoundError: If the file does not exist.
    Exception: If there is an error while loading the pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"Data successfully loaded from {file_path}")
            return data
    except Exception as e:
        print(f"An error occurred while loading data from {file_path}: {e}")
        return None


def save_to_pickle(data, pickle_file_path):
    """
    Save data to a pickle file.

    Parameters:
    data: The data to be pickled (e.g., a list of dataframes).
    pickle_file_path (str): The path to the output pickle file.
    """
    try:
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Data successfully saved to {pickle_file_path}")
    except Exception as e:
        print(f"An error occurred while saving data to pickle: {e}")
