from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
from sentence_transformers import SentenceTransformer
from auxiliar_functions.general_purpose_functions import load_pickle, save_to_pickle


# Function to create sentence embeddings
def create_sentence_embeddings(df, sentence_model):
    """
    Create sentence embeddings for each sentence in the DataFrame using a provided SentenceTransformer model.

    Args:
        df (pd.DataFrame): DataFrame containing sentences to embed. It should have a column named 'data' containing 
                           the sentences to encode.
        sentence_model (SentenceTransformer): Pretrained SentenceTransformer model used for encoding sentences.

    Returns:
        torch.Tensor: Torch tensor containing normalized embeddings of the sentences from the DataFrame.
                      Each row corresponds to the embedding of a sentence, normalized to have unit length.
    """

    df['col_embedding'] = None

    # Iterate over each row in the DataFrame
    for i, sentence in tqdm(enumerate(df['data']), desc='Encoding Columns'):
        # Encode the sentence using the provided sentence model
        embedding = sentence_model.encode(sentence, convert_to_tensor=True, normalize_embeddings=False)
        
        # Assign the embedding to the 'col_embedding' column in the DataFrame
        df.at[i, 'col_embedding'] = embedding

    # Step 2: Convert embeddings to Torch Tensor
    num_elements = len(df['col_embedding'])
    tensor_size = len(df['col_embedding'][0])
    col_embeddings_tensor = torch.empty((num_elements, tensor_size))

    for i, element in enumerate(tqdm(df['col_embedding'], desc='Converting to Torch Tensor')):
        col_embeddings_tensor[i, :] = torch.tensor(element)

    # Normalize the rows to have unit length (optional but often recommended)
    tensor_normalized = F.normalize(col_embeddings_tensor, p=2, dim=1)

    return tensor_normalized


def main(pretrained_model, file_name):
    
    file_to_save = 'data/embeddings/' + pretrained_model + '_tensors.pkl'
    data_path = 'data/structured_data/' + file_name    
    dataset = load_pickle(data_path)

    df_train = dataset['train']
    df_test = dataset['test']
    df_dev = dataset['dev']

    sentence_model = SentenceTransformer(pretrained_model)

    train_loaded_tensor = create_sentence_embeddings(df_train, sentence_model)
    test_loaded_tensor = create_sentence_embeddings(df_test, sentence_model)
    dev_loaded_tensor = create_sentence_embeddings(df_dev, sentence_model)

    # Create a dictionary to store the DataFrames
    tensors = {'train':train_loaded_tensor, 'dev': dev_loaded_tensor, 'test': test_loaded_tensor}

    # Save the dictionary of DataFrames to a pickle file
    save_to_pickle(tensors, file_to_save)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process data and save sentence embeddings.')
    parser.add_argument('--pretrained_model', type=str, default='all-distilroberta-v1', help='Name of the pretrained SentenceTransformer model')
    parser.add_argument('--structured_data_file', type=str, default='sotab_data_preprocessed.pkl', help='File name of the structured data pickle file.')
    
    # Call main function with parsed arguments
    args = parser.parse_args()
    main(args.pretrained_model, args.structured_data_file)

    