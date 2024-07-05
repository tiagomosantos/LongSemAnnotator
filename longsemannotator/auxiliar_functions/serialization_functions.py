import torch

def serialize_target_col(target_col_tokenized, tokenizer):

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    tokenized_input = []

    tokenized_input.append(cls_token_id)  # Add [CLS] token

    if len(target_col_tokenized) > 255:  # w/2 +1 - [CLS] - [SEP] = 256 + 1 - 1 - 1 = 255
        target_col_tokenized = target_col_tokenized[0:255]
    else:
        target_col_tokenized.extend([pad_token_id] * (255 - len(target_col_tokenized)))  # Pad the target column

    tokenized_input.extend(target_col_tokenized)  # Add target column
    tokenized_input.append(sep_token_id)  # Add [SEP] token

    return tokenized_input    

def serialize_related_columns(df, tokenizer, tokenized_input, related_cols_ids):

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    related_table_cols_tokenized = []

    tokenized_input.append(cls_token_id)  # Add [CLS] token

    for related_col_id in related_cols_ids:
        related_table_col_tokenized = df['tokenized_column_64'].loc[related_col_id]
        related_table_cols_tokenized.extend(related_table_col_tokenized)

    if len(related_table_cols_tokenized) > 255:  # w/2 +1 - [CLS] - [SEP] = 256 + 1 - 1 - 1 = 255
        related_table_cols_tokenized = related_table_cols_tokenized[0:255]
    else:
        related_table_cols_tokenized.extend(
            [pad_token_id] * (255 - len(related_table_cols_tokenized)))  # Pad the Related columns

    tokenized_input.extend(related_table_cols_tokenized)  # Add Related columns
    tokenized_input.append(sep_token_id)  # Add [SEP] token


def serialize_dataframe(row, df_train, tokenizer, num_subsets=4, num_related_cols=4):
    """
    Tokenize a column and its related columns in a DataFrame row using a tokenizer.

    Args:
        row (pd.Series): A row from the DataFrame containing columns to tokenize.
        df_train (pd.DataFrame): DataFrame containing the training data.
        tokenizer (transformers.AutoTokenizer): Pretrained tokenizer from the Hugging Face transformers library.
        num_subsets (int, optional): Number of groups of related columns to tokenize. Defaults to 4.
        num_related_cols (int, optional): Number of related columns to tokenize. Defaults to 4.

    Returns:
        torch.Tensor: LongTensor containing the token IDs of the tokenized input.

    """
    pad_token_id = tokenizer.pad_token_id

    # Tokenize the target column
    tokenized_input = serialize_target_col(row['tokenized_column_256'], tokenizer)

    # Add 256 tokens with no attention (local windows /2)
    tokenized_input.extend([pad_token_id] * (256))

    # Ensure there are enough related columns to create the specified number of subsets
    total_related_needed = num_subsets * num_related_cols
    if len(row['closest_indexes']) < total_related_needed:
        raise ValueError(f"Not enough related columns: required {total_related_needed}, found {len(row['closest_indexes'])}")

    # List of related columns groups
    related_columns = [row['closest_indexes'][i:i+num_related_cols] for i in range(0, len(row['closest_indexes']), num_related_cols)[:num_subsets]]

    # Iterate over related columns and tokenize
    for cols in related_columns:
        serialize_related_columns(df_train, tokenizer, tokenized_input, cols)
        # Add 256 tokens with no attention (local windows /2)
        tokenized_input.extend([pad_token_id] * (256))

    token_ids_cta = torch.LongTensor(tokenized_input) # Convert to LongTensor

    return token_ids_cta