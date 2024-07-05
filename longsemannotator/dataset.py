import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from auxiliar_functions.general_purpose_functions import load_pickle
import os

def collate_fn(samples):
    PADDING_VALUE = 1  # Longformer padding value
    CLS_TOKEN_ID = 0

    data = pad_sequence([sample["input_ids"] for sample in samples],
                        batch_first=True,
                        padding_value=PADDING_VALUE)

    label = torch.stack([sample["labels"] for sample in samples])

    # Generate attention mask
    attention_mask = (data != PADDING_VALUE).int()

    # Generate global attention mask
    # Setting global attention to all 'cls' tokens
    global_attention_mask = (data == CLS_TOKEN_ID).int()
    
    batch = {
        "input_ids": data,
        "labels": label,
        "attention_mask": attention_mask,
        "global_attention_mask": global_attention_mask,
    }

    return batch

class SOTABDataset(Dataset):
    def __init__(self, data, mlb):
        self.data = data
        self.mlb = mlb
        self.num_classes = len(self.mlb.classes_)
 
    def get_num_classes(self):
        return len(self.mlb.classes_)

    def __len__(self):
        return len(self.data)
    
    def compute_class_weights(self):
        # Initialize dictionary to store class occurrences
        class_occurrences = {idx: 0 for idx in range(self.num_classes)}

        # Iterate through each sample's label array and count occurrences
        for labels in self.data['cta_labels_encoded']:
            for idx, label in enumerate(labels):
                if label == 1:
                    class_occurrences[idx] += 1

        # Compute weights based on occurrences
        total_samples = len(self.data)

        class_weights = [total_samples / count if count != 0 else 0 for count in class_occurrences.values()]

        return torch.tensor([class_weights])

    def __getitem__(self, idx):
        data = self.data.iloc[idx].copy()
        
        return {
            'input_ids': data['data_tensor'],
            'labels': data['cta_labels_encoded'],
        }
    

# Function to encode labels
def encode_labels(labels, lb):
    return torch.tensor(lb.transform([labels])[0])


def load_dataset(folder, dataset_file, lb_file, train=True):
    
    dataset_file_path = os.path.join(folder, dataset_file)
    dataset = load_pickle(dataset_file_path)

    lb_file_path = os.path.join(folder, lb_file)
    lb = load_pickle(lb_file_path)

    if train:
        train_data = dataset['train']
        train_data['cta_labels_encoded'] = train_data['label'].apply(lambda x: encode_labels(x, lb))
        train_data = train_data.drop(columns='label')
        train_dataset = SOTABDataset(train_data, lb)

        valid_data = dataset['dev']
        valid_data['cta_labels_encoded'] = valid_data['label'].apply(lambda x: encode_labels(x, lb))
        valid_data = valid_data.drop(columns='label')
        valid_dataset = SOTABDataset(valid_data, lb)

        return train_dataset, valid_dataset
    
    else:
        test_data = dataset['test']
        test_data['cta_labels_encoded'] = test_data['label'].apply(lambda x: encode_labels(x, lb))
        test_data = test_data.drop(columns='label')
        test_dataset = SOTABDataset(test_data, lb)

        return test_dataset
