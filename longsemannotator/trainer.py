import numpy as np
import evaluate
from transformers import Trainer,TrainingArguments
from base_model import prepare_model
from dataset import collate_fn, load_dataset
import argparse
import time
import os

import wandb
WANDB_API_KEY = 'b0161826f2557200dc5cff4013ba25fd69fdb64a'
wandb.login(key=WANDB_API_KEY, force=True)

def compute_metrics(pred):
    
    labels = pred.label_ids
    preds = pred.predictions

    labels = np.argmax(labels, axis=1)
    preds = np.argmax(preds, axis=1)

    metric1 = evaluate.load("f1")
    metric2 = evaluate.load("f1")
    metric3 = evaluate.load("f1")

    micro_f1 = metric1.compute(predictions=preds, references=labels, average='micro')['f1']
    macro_f1 = metric2.compute(predictions=preds, references=labels, average='macro')['f1']
    weighted_f1 = metric3.compute(predictions=preds, references=labels, average='weighted')['f1']

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

def main(dataset_file, lb_file, num_subsets=4, num_classes=91):

    path = 'data/ready_to_model_data'
    print(f"Loading dataset from {path}/{dataset_file} and label binarizer from {path}/{lb_file}")
    train_dataset, valid_dataset = load_dataset(path, dataset_file, lb_file, True)
    print("Dataset loaded successfully.")
    
    print("Preparing model...")
    model = prepare_model(num_classes, num_subsets)
    print("Model prepared successfully.")

    gradient_accumulation_steps = 4
    learning_rate = 3.5e-5
    warmup_ratio = 0.05

    checkpoints_folder = 'checkpoints/'
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_dir = checkpoints_folder+current_time
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=1,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=3,
        logging_strategy='steps',
        logging_steps=10,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        save_strategy='steps',
        save_steps=1359,
        save_total_limit=5,
        seed=0,
        fp16=False,
        eval_steps=1359,
        run_name='sotab_closest_8',
        optim='adamw_torch',
        report_to='wandb'
    )

    # define training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # start training loop
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the LongSemAnnotator model.")
    parser.add_argument('--dataset_file', type=str, default='all-distilroberta-v1_closest_16.pkl', help='Dataset file name.')
    parser.add_argument('--lb_file', type=str, default='sotab_lb.pkl', help='Label Binarizer file name.')
    parser.add_argument('--num_subsets', type=int, default=4, help='Number of subsets of related columns.')
    parser.add_argument('--num_classes', type=int, default=91, help='Total number of classes to classify.')

    # Call main function with parsed arguments
    args = parser.parse_args()
    main(args.dataset_file, args.lb_file, args.num_subsets, args.num_classes)
