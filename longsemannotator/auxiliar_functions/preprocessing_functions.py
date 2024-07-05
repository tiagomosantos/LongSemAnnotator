import re
import pandas as pd
from tqdm import tqdm
import os 

pd.set_option('future.no_silent_downcasting', True)

def is_http_url(string):
    """
    Check if a given string is a valid HTTP or HTTPS URL.

    Args:
        string (str): The string to be checked.

    Returns:
        bool: True if the string is a valid HTTP/HTTPS URL, False otherwise.
    """
    # Regular expression pattern for HTTP URL
    http_pattern = re.compile(r'^(?:https?:)?//(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,6}(?:/[^/#?]+)+/?$')
    
    # Check if the string matches the pattern
    if http_pattern.match(string):
        return True
    else:
        return False

def is_schema_org_url(string):
    """
    Check if a given string is a URL from the schema.org domain.

    Args:
        string (str): The string to be checked.

    Returns:
        bool: True if the string is a schema.org URL, False otherwise.
    """
    # Regular expression pattern for schema.org URL
    schema_org_pattern = re.compile(r'^https://schema\.org/.*$')
    schema_org_pattern2 = re.compile(r'^http://schema\.org/.*$')

    # Check if the string matches either pattern
    if schema_org_pattern.match(string) or schema_org_pattern2.match(string):
        return True
    else:
        return False

def apply_preprocess_to_str(text, table_labels, index):
    """
    Preprocess a string by removing backslashes and quotes.

    Parameters:
    text (str): The input string to be preprocessed.

    Returns:
    str: The preprocessed string.
    """
    try:
        if isinstance(text, (str, bytes)):

            if is_http_url(text):
                label = table_labels.loc[index]['label']
                # Check if an empty space is present in the URL
                if ' ' in text:
                    # If it ends with .png, .jpg, .JPG, .jpeg or .mp3 is a URL to a file
                    if text.endswith('.png') or text.endswith('.jpg') or text.endswith('.jpeg') or text.endswith('.JPG') or text.endswith('.mp3'):
                        # Get what is before the fist empty space
                        selected_text = text[:text.find(" ")]
                        return selected_text
                    # It is not an URL
                    else:
                        # Get the position of the first empty char
                        first_empty_char_position = text.find(" ")
                        # Get what is after the position of the first empty char
                        string_after_first_empty_char = text[first_empty_char_position + 1:]
                        return string_after_first_empty_char
                else:
                    # If it is a schema.org URL
                    if is_schema_org_url(text):
                        last_slash_position = text.rfind("/")
                        string_after_last_slash = text[last_slash_position + 1:]
                        return string_after_last_slash
                    # It is really an URL
                    else: 
                        return text
            else:
                return re.sub(r'[\"\\]', '', text)
        elif text is None:
            return ''
        else:
            return str(text)  # Convert non-string/bytes-like input to string
    except Exception as e:
        print(f"Error occurred while processing text: {text}")
        raise e

def apply_preprocess_to_list(col, table_labels, index):
    """
    Preprocess each element of a list and concatenate them with spaces.

    Parameters:
    col (list): The input list to be preprocessed.

    Returns:
    str: The concatenated preprocessed text.
    """
    preprocessed_texts = [apply_preprocess_to_str(text, table_labels, index) for text in col]
    return ' '.join(preprocessed_texts)

def process_columns(col, table_labels, index):
    """
    Process a column of data based on its type.

    Parameters:
    col: The input column to be processed.

    Returns:
    str: The preprocessed text.
    """
    if isinstance(col, (int, float)):
        return apply_preprocess_to_str(str(col), table_labels, index)
    elif isinstance(col, list):
        return apply_preprocess_to_list(col, table_labels, index)
    elif isinstance(col, str):
        return apply_preprocess_to_str(col, table_labels, index)
    else:
        raise ValueError(f"Unsupported data type: {type(col)}")

def concatenate_columns(row, table_labels):
    """
    Concatenate preprocessed texts from each element of a row.

    Parameters:
    row: The input row of data.

    Returns:
    str: The concatenated preprocessed text.
    """
    row = row.fillna('')
    index = row.name
    return ' '.join(row.apply(process_columns, args=(table_labels, index)))

def convert_json_to_df(test_path, data_file, labels_df):
    """
    Convert JSON data to a structured DataFrame.

    Parameters:
    test_path (str): Path to the directory containing JSON data.
    data_file (str): Name of the JSON data file.
    csv_path (str): Path to the directory containing CSV label files.
    csv_file (str): Name of the CSV label file.

    Returns:
    pd.DataFrame: The structured DataFrame containing preprocessed data.
    """
    data_df = pd.read_json(test_path + data_file, compression='gzip', lines=True)
    table_labels = labels_df[labels_df['table_name'] == data_file]

    selected_columns = table_labels.merge(data_df.T, left_on='column_index', right_index=True)
    new_df = selected_columns[['table_name', 'column_index', 'label']].copy()
    selected_columns.drop(columns=['table_name', 'column_index', 'label'], inplace=True)
    check_columns = pd.to_numeric(selected_columns.columns, errors='coerce')
    all_integers = all(isinstance(col, int) for col in check_columns)

    if all_integers:
        new_df['data'] = selected_columns.apply(concatenate_columns, axis=1, args=(table_labels,))
        return new_df
    else:
        raise ValueError("Not all columns are integers")
 
def preprocess_sotab_split(directory, csv_file):
    """
    Preprocess the files in the specified directory using the given CSV file for labels.

    Args:
        directory (str): Directory where the files are located.
        csv_file (str): Path to the CSV file containing the labels.

    Returns:
        pd.DataFrame: The concatenated DataFrame containing the processed data from all files.
    """
    # Get a list of all files in the directory that end with '.json.gz'
    files = [f for f in os.listdir(directory) if f.endswith('.json.gz')]

    # Read the CSV file into a DataFrame
    labels_df = pd.read_csv(csv_file)

    # New DataFrame to store the processed data
    new_df = pd.DataFrame()
    
    # Use tqdm to create a progress bar
    with tqdm(total=len(files), desc='Processing files') as pbar:
        # Iterate over each file
        for file in files:
            # Apply the function to extract data from the file
            df = convert_json_to_df(directory, file, labels_df)

            # Concatenate the result with train_df
            new_df = pd.concat([new_df, df], ignore_index=True)

            # Update the progress bar
            pbar.update(1)
    
    return new_df
