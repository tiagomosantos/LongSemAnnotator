import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset, collate_fn
from base_model import LongformerForMultiOutputClassification
import pandas as pd
from sklearn import metrics
import os 
from auxiliar_functions.general_purpose_functions import load_pickle
import argparse

def main(dataset_file, lb_file, checkpoint_path, batch_size=4):

    path = 'data/ready_to_model_data'
    test_dataset = load_dataset(path, dataset_file, lb_file, False)

    test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_folder = "checkpoints/"
    epoch_folder = checkpoint_path
    checkpoint_path = checkpoint_folder + epoch_folder
    model = LongformerForMultiOutputClassification.from_pretrained(checkpoint_path)    
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    desc  = 'Validating...'
    progress_bar_val = tqdm(range(len(test_dataloader)), desc=desc, position=0, mininterval=10)
    for batch in test_dataloader:
        data = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        global_attention_mask = batch["global_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=data, attention_mask=attention_mask,
                            global_attention_mask=global_attention_mask, labels=labels)

        logits = outputs['logits']

        references = torch.argmax(labels, dim=1)
        predictions = torch.argmax(logits, dim=1)

        y_pred.extend(predictions.tolist())
        y_true.extend(references.tolist())

        progress_bar_val.update(1)

    progress_bar_val.close()   

    lb_file_path = os.path.join(path, lb_file)
    lb = load_pickle(lb_file_path)

    results = metrics.classification_report(y_true, y_pred, digits=2, zero_division=0.0, output_dict=True, target_names=lb.classes_)

    df = pd.DataFrame(results).transpose()
    # save dataframe as csv file
    df.to_csv('results/' + checkpoint_path + '.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on a test dataset.')
    parser.add_argument('--dataset_file', type=str, default='all-distilroberta-v1_closest_16.pkl', help='Dataset file name.')
    parser.add_argument('--lb_file', type=str, default='sotab_lb.pkl', help='Label Binarizer file name.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='The path to the model checkpoint directory.')
    parser.add_argument('--batch_size', type=int, default=4, help='The batch size for evaluation.')

    args = parser.parse_args()
    main(args.dataset_file, args.lb_file, args.checkpoint_path, args.batch_size)