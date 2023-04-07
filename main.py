import os
import logging
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, AutoTokenizer

from data import Dataset, load_data, prepare_data

MODELS = {
    "BERT": "bert-base-uncased",
    "ROBERTA": "roberta-base",
    "DEBERTA": "microsoft/deberta-base",
    "DISTILBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2"}

VALID_DATASETS = ['LIAR', 'FakeTrue']
VALID_MODELS = list(MODELS.keys())

def compute_metrics(pred):
    lbls = pred.label_ids
    prds = pred.predictions.argmax(-1)
    acc = accuracy_score(lbls, prds)
    return {"acc": acc}


def train_transformer(model, dataset, output_dir, training_batch_size, eval_batch_size, learning_rate,
                      num_train_epochs, weight_decay,
                      disable_tqdm=False):

    # training params
    model_ckpt = MODELS[model]
    print(model_ckpt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/{model_ckpt}-finetuned-{dataset['name']}"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    
    train_data, test_data, label_dict = prepare_data(dataset, tokenizer, Dataset)

    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=len(label_dict)).to(device)
    logging_steps = len(train_data) // training_batch_size

    training_args = TrainingArguments(output_dir=output,
                                          num_train_epochs=num_train_epochs,
                                          learning_rate=learning_rate,
                                          per_device_train_batch_size=training_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          weight_decay=weight_decay,
                                          evaluation_strategy="epoch",
                                          disable_tqdm=disable_tqdm)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_data,
                      eval_dataset=test_data,
                      compute_metrics=compute_metrics,
                      tokenizer=tokenizer)

    trainer.train()

    evaluate_trainer(trainer, test_data, output_dir)

    # save model
    model.save_pretrained(f"{output}/model")


    def evaluate_trainer(trainer, test_data, output_dir):
        # accuracy
        prediction_output = trainer.predict(test_data)
        logging.info(f"Prediction metrics: {prediction_output.metrics}")

        # confusion matrix
        y_preds = np.argmax(prediction_output.predictions, axis=1)
        y_true = prediction_output.label_ids
        cm = confusion_matrix(y_true, y_preds)
        logging.info(f"Confusion matrix:\n{cm}")

        # create file if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # save results to file
        with open(f"{output_dir}/eval_results.json", "a") as f:
            f.write("\n")
            json.dump(prediction_output.metrics, f)


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run text classification on the given dataset with the given model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('dataset', type=str, choices=VALID_DATASETS, help='Dataset to use.')
    parser.add_argument('model', type=str, choices=VALID_MODELS, help='Model to use.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory.')
    parser.add_argument('--log_level', type=str, default='info', help='Log level.')
    parser.add_argument('--log_to_file', action='store_true', help='Log to file.')
    parser.add_argument('--log_file', type=str, default='log.txt', help='Log file.')

    parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--weight_decay', type=float, default=0.00, help='Weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout.')

    args = parser.parse_args()


    config = vars(args)

    logging.info("Starting...")
    logging.debug("Arguments: %s", args)

    # Start training
    logging.info(f"Loading {args.dataset} data...")
    dataset = load_data(args.dataset)


    train_transformer(config["model"],
                          dataset,
                          config["output_dir"],
                          training_batch_size=config["batch_size"],
                          eval_batch_size=config["batch_size"],
                          learning_rate=config["learning_rate"],
                          num_train_epochs=config["num_train_epochs"],
                          weight_decay=config["weight_decay"])


if __name__ == '__main__':
    main()