from auxiliar_functions.general_purpose_functions import load_pickle, save_to_pickle
import pandas as pd
from auxiliar_functions.filtering_functions import create_faiss_index, search_faiss_index, add_closest_indexes_to_df
import faiss
import numpy as np
from tqdm import tqdm
from auxiliar_functions.tokenization_functions import tokenize_dataframe
from auxiliar_functions.serialization_functions import serialize_dataframe
from transformers import AutoTokenizer
import argparse

def filtering_small_dataset(pretrained_model, structured_data_file):

    # Load the tensors
    tensors_path = 'data/embeddings/' + pretrained_model + '_tensors.pkl'
    loaded_tensors = load_pickle(tensors_path)
    train_tensors, test_tensors, dev_tensors = loaded_tensors['train'], loaded_tensors['test'], loaded_tensors['dev']

    # Load the structured data
    structured_data_path = 'data/structured_data/' + structured_data_file
    dataset = load_pickle(structured_data_path)
    df_train, df_test, df_dev = dataset['train'], dataset['test'], dataset['dev']

    # Add column to store embeddings
    df_train.insert(len(df_train.columns), 'col_embedding', None)
    df_test.insert(len(df_test.columns), 'col_embedding', None)
    df_dev.insert(len(df_dev.columns), 'col_embedding', None)

    for idx, _ in enumerate(range(df_train.shape[0])):
        df_train.at[idx, 'col_embedding'] = train_tensors[idx]

    for idx, _ in enumerate(range(df_test.shape[0])):
        df_test.at[idx, 'col_embedding'] = test_tensors[idx]

    for idx, _ in enumerate(range(df_dev.shape[0])):
        df_dev.at[idx, 'col_embedding'] = dev_tensors[idx]

    df_small_training = pd.read_csv('data/raw_data/CTA_training_small_gt.csv')

    # MERGE SMALL TRAINING SET -----------------------------------
    df_small = pd.merge(df_train, df_small_training, on=['column_index','table_name','label'], how='inner')

    # END OF MERGE SMALL TRAINING SET -----------------------------------

    # FILTERING ------------------------------------------
    print("Filtering data...")

    # Use the small training set to create the FAISS index
    train_tensors = df_small['col_embedding'].to_list()
    train_tensors = np.array(train_tensors)

    # Create and populate FAISS index
    embeddings_dimension = train_tensors.shape[1]
    faiss_index = create_faiss_index(train_tensors, embeddings_dimension, faiss.METRIC_INNER_PRODUCT)

    # Search and add closest indexes for train, dev, and test sets
    print("Searching closest indexes to Train...")
    _, train_indexes = search_faiss_index(faiss_index, train_tensors, 21)
    add_closest_indexes_to_df(df_train, train_indexes[:, 1:21])

    print("Searching closest indexes to Dev...")
    _, dev_indexes = search_faiss_index(faiss_index, dev_tensors, 20)
    add_closest_indexes_to_df(df_dev, dev_indexes)

    print("Searching closest indexes to Test...")
    _, test_indexes = search_faiss_index(faiss_index, test_tensors, 20)
    add_closest_indexes_to_df(df_test, test_indexes)    

    # END OF FILTERING -----------------------------------
    
    # Clean the unnecessary columns
    df_small.drop(columns=['col_embedding'], inplace=True)
    df_dev.drop(columns=['col_embedding'], inplace=True)
    df_test.drop(columns=['col_embedding'], inplace=True)


    return df_small, df_dev, df_test

def generate_small_dataset(pretrained_model, structured_data_file, num_subsets=4, num_related_cols=4):
    """
    Main function to process data, create FAISS index, and serialize results.
    
    Args:
        pretrained_model (str): Name of the pretrained SentenceTransformer model.
        structured_data_file (str): Path to the structured data pickle file.
        num_subsets (int): Number of groups of related columns to tokenize. Defaults to 4.
        num_related_cols (int): Number of related columns to tokenize per subset. Defaults to 4.
    """

    df_train, df_dev, df_test = filtering_small_dataset(pretrained_model, structured_data_file)

    # TOKENIZATION ----------------------------------------
    print("Tokenizing data...")
    # Define the tokenizer
    tokenizer_name = 'allenai/longformer-base-4096'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize data
    df_train = tokenize_dataframe(df_train, tokenizer, "Tokenizing CTA Training Columns")
    df_dev = tokenize_dataframe(df_dev, tokenizer, "Tokenizing CTA Validation Columns")
    df_test = tokenize_dataframe(df_test, tokenizer, "Tokenizing CTA Test Columns")

    # END OF TOKENIZATION ---------------------------------

    # SERIALIZATION ----------------------------------------
    print("Serializing data...")
    # Serialize the data    
    tqdm.pandas(desc="Tokenizing CTA Training Columns", total=len(df_train))
    df_train["data_tensor"] = df_train.progress_apply(serialize_dataframe, args=(df_train, tokenizer, num_subsets, num_related_cols), axis=1)

    tqdm.pandas(desc="Tokenizing CTA Validation Columns", total=len(df_dev))
    df_dev["data_tensor"] = df_dev.progress_apply(serialize_dataframe, args=(df_train, tokenizer, num_subsets, num_related_cols), axis=1)

    tqdm.pandas(desc="Tokenizing CTA Test Columns", total=len(df_test))
    df_test["data_tensor"] = df_test.progress_apply(serialize_dataframe, args=(df_train, tokenizer, num_subsets, num_related_cols), axis=1)

    # END OF SERIALIZATION --------------------------------

    # Drop unnecessary columns
    drop_columns = ['data', 'closest_indexes', 'tokenized_column_64', 'tokenized_column_256']
    df_train = df_train.drop(columns=drop_columns)
    df_dev = df_dev.drop(columns=drop_columns)
    df_test = df_test.drop(columns=drop_columns)

    # Create a dictionary to store the DataFrames
    dataframes = {'train': df_train, 'dev': df_dev, 'test': df_test}

    # Save the dictionary of DataFrames to a pickle file
    tensors_path = 'data/embeddings/' + pretrained_model + '_tensors.pkl'
    embeddings_model_name = tensors_path.split('/')[-1].split('_')[0]
    total_related_needed = num_subsets * num_related_cols
    pickle_file = f"{embeddings_model_name}_closest_{total_related_needed}.pkl"
    pickle_file_path = f"data/ready_to_model_data/small_{pickle_file}"

    save_to_pickle(dataframes, pickle_file_path)


def generate_special_cases(structured_data_file, pretrained_model, num_subsets=4, num_related_cols=4):
    # Load the structured data
    structured_data_path = 'data/structured_data/' + structured_data_file
    dataset = load_pickle(structured_data_path)
    df_test = dataset['test']

    df_corner_cases = pd.read_csv('data/raw_data/CTA_test_corner_cases_gt.csv')
    df_format_heterogeneous = pd.read_csv('data/raw_data/CTA_test_format_heterogeneity_gt.csv')
    df_missing_values = pd.read_csv('data/raw_data/CTA_test_missing_values_gt.csv')
    df_random_values = pd.read_csv('data/raw_data/CTA_test_random_gt.csv')

    df_corner_cases = pd.merge(df_test, df_corner_cases, on=['column_index','table_name'], how='inner')
    df_format_heterogeneous = pd.merge(df_test, df_format_heterogeneous, on=['column_index','table_name'], how='inner')
    df_missing_values = pd.merge(df_test, df_missing_values, on=['column_index','table_name'], how='inner')
    df_random_values = pd.merge(df_test, df_random_values, on=['column_index','table_name'], how='inner')

    df_corner_cases.rename(columns={"label_x": "label"}, inplace=True)
    df_format_heterogeneous.rename(columns={"label_x": "label"}, inplace=True)
    df_missing_values.rename(columns={"label_x": "label"}, inplace=True)
    df_random_values.rename(columns={"label_x": "label"}, inplace=True)

    df_corner_cases.drop(columns=['label_y'], inplace=True)
    df_format_heterogeneous.drop(columns=['label_y'], inplace=True)
    df_missing_values.drop(columns=['label_y'], inplace=True)
    df_random_values.drop(columns=['label_y'], inplace=True)

    # Create a dictionary to store the DataFrames
    dataframes = {'corner':df_corner_cases, 'hetero': df_format_heterogeneous, 'missing': df_missing_values, 'random':df_random_values}

    # Specify the path where you want to save the pickle file
    tensors_path = 'data/embeddings/' + pretrained_model + '_tensors.pkl'
    embeddings_model_name = tensors_path.split('/')[-1].split('_')[0]
    total_related_needed = num_subsets * num_related_cols
    pickle_file = f"{embeddings_model_name}_closest_{total_related_needed}.pkl"
    pickle_file_path = f"data/ready_to_model_data/special_cases_{pickle_file}"

    save_to_pickle(dataframes, pickle_file_path)

def main (pretrained_model, structured_data_file, num_subsets=4, num_related_cols=4):
    """
    Main function to process data, create FAISS index, and serialize results.
    
    Args:
        pretrained_model (str): Name of the pretrained SentenceTransformer model.
        structured_data_file (str): Path to the structured data pickle file.
        num_subsets (int): Number of groups of related columns to tokenize. Defaults to 4.
        num_related_cols (int): Number of related columns to tokenize per subset. Defaults to 4.
    """
    generate_small_dataset(pretrained_model, structured_data_file, num_subsets, num_related_cols)
    generate_special_cases(pretrained_model, structured_data_file, num_subsets, num_related_cols)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Tokenize and serialize DataFrame columns.")
    parser.add_argument('--pretrained_model', type=str, default='all-distilroberta-v1', help='Name of the pretrained SentenceTransformer model.')
    parser.add_argument('--structured_data_file', type=str, default='sotab_data_preprocessed.pkl', help='File name of the structured data pickle file.')
    parser.add_argument('--num_subsets', type=int, default=4, help='Number of groups of related columns to tokenize.')
    parser.add_argument('--num_related_cols', type=int, default=4, help='Number of related columns to tokenize per subset.')
    
    # Call main function with parsed arguments
    
    # Call main function with parsed arguments
    args = parser.parse_args()
    main(args.pretrained_model, args.structured_data_file, args.num_subsets, args.num_related_cols)