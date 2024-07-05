from tqdm import tqdm

def tokenize_text(tokenizer, text, max_length):
    """
    Tokenizes text using a specified maximum length and returns input_ids.

    Args:
        text (str): Input text to tokenize.
        max_length (int): Maximum length of the tokenized sequence.

    Returns:
        list: List of input_ids representing the tokenized text.
    """
    return tokenizer(text[:2000], add_special_tokens=False, truncation=True, max_length=max_length)['input_ids']

def tokenize_dataframe(df, tokenizer, desc):
    """
    Tokenizes each row in a DataFrame and adds tokenized columns.

    Args:
        df (pd.DataFrame): DataFrame containing text data to tokenize.
        tokenizer (AutoTokenizer): Pretrained tokenizer from transformers library.
        desc (str): Description for tqdm progress bar.

    Returns:
        pd.DataFrame: Updated DataFrame with tokenized columns.
    """
    tqdm.pandas(desc=desc+' - 64 tokens')
    df['tokenized_column_64'] = df['data'].progress_apply(lambda x: tokenize_text(tokenizer, x, 63))
    tqdm.pandas(desc=desc+' - 256 tokens')
    df['tokenized_column_256'] = df['data'].progress_apply(lambda x: tokenize_text(tokenizer, x, 254))
    return df
