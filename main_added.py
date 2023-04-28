'''
The following sources were utilized in the development of this project:
https://github.com/FKarl/short-text-classification
https://github.com/lgalke/text-clf-baselines
'''

import os
from tqdm import tqdm, trange
import joblib
import logging
import json
import numpy as np

from data import Dataset, Dataset2, load_data, prepare_data

from scipy.stats import randint as sp_randint
from scipy.stats import loguniform

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import tokenizers

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, Trainer, \
    TrainingArguments, AutoTokenizer


MODELS = {
    "BERT": "bert-base-uncased",
    "ROBERTA": "roberta-base",
    "DEBERTA": "microsoft/deberta-base",
    "DISTILBERT": "distilbert-base-uncased",
    "ALBERT": "albert-base-v2"}

VALID_DATASETS = ['LIAR', 'FakeTrue']
VALID_MODELS = list(MODELS.keys())
VALID_MODELS = ["SVM", "LSTM","RandomForest", "NaiveBayes", "LogisticRegression"] + list(MODELS.keys())


def compute_metrics(pred):
    lbls = pred.label_ids
    prds = pred.predictions.argmax(-1)
    acc = accuracy_score(lbls, prds)
    return {"acc": acc}

def train_transformer(model, dataset, output_dir, training_batch_size, eval_batch_size, learning_rate, 
                      num_train_epochs, weight_decay, disable_tqdm=False):
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

    evaluate_transformer(trainer, test_data, output_dir)

    # save model
    model.save_pretrained(f"{output}/model")

class MLTrainer:
    def __init__(self, model, model_path, label_dict):
        self.model = model
        self.model_path = model_path
        self.label_dict = label_dict

    def predict(self, text):
        predictions = self.model.predict(text)
        return predictions

    def save_model(self, output_dir):
        save_ml_model(self.model, self.model_path, output_dir)


def train_ML(dataset, model, output_dir):
    # Prepare data
    train_data, test_data, label_dict = prepare_data(dataset, None, Dataset)

    train_text = [train_data.text['input_ids'][i] for i in range(len(train_data))]
    train_labels = [train_data.labels[i] for i in range(len(train_data))]

    if model == "SVM":
        # Train SVM
        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            SVC(kernel="linear")
            )
    elif model == "NaiveBayes":
        parameters = {'alpha':[0.00001,0.0005, 0.0001,0.005,0.001,0.05,0.01,0.1,0.5,1,5,10,50,100]}
        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            RandomizedSearchCV(MultinomialNB(class_prior=[0.5, 0.5]), parameters,n_jobs = -1, cv= 5, scoring='roc_auc')
            )
    elif model == "RandomForest":
        param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators":[10, 20,  30,  40, 50, 80,100,150, 200]}

        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=20)
            )

    elif model == "LogisticRegression":
        space = dict()
        space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
        space['penalty'] = ['none', 'l1', 'l2']
        space['C'] = loguniform(1e-5, 100)

        pipeline = make_pipeline(
            TfidfVectorizer(),
            StandardScaler(with_mean=False),
            RandomizedSearchCV(LogisticRegression(), space, n_iter=20, scoring='accuracy', n_jobs=-1, cv=5, random_state=1)
            )

    pipeline.fit(train_text, train_labels)

    # Create SVMTrainer object
    ml_trainer = MLTrainer(pipeline, model, dataset["label_dict"])

    # Evaluate using evaluate_trainer
    evaluate_ml(ml_trainer, test_data, output_dir)

    # Save model
    ml_trainer.save_model(output_dir)

def save_ml_model(model, model_name, output_dir):
    model_path = os.path.join(output_dir, model_name+"_model.joblib")
    joblib.dump(model, model_path)
    

class LSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, bidirectional, hidden_size, num_layers,
                 dropout):
        super(LSTM, self).__init__()

        self.input_size = vocab_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embed = nn.EmbeddingBag(vocab_size, hidden_size)
        embedding_dim = hidden_size

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)

        # LSTM architecture
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

        # linear layer on top
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, text, offset, label):
        # embedding
        embedded = self.embedding(text, offset)
        embedded = self.dropout(embedded)

        # lstm
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # linear layer
        out = self.fc(lstm_out)

        # loss
        loss = self.loss_function(out, label)
        return loss, out


def collate_for_lstm(list_of_samples):
    """
    Collate function that creates batches of flat docs tensor and offsets
    """
    offset = 0
    flat_docs, offsets, labels = [], [], []
    for doc, label in list_of_samples:
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        labels.append(label)
        offset += len(doc)
    return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)

def train_lstm(dataset, output_dir, epochs, warmup_steps, learning_rate, weight_decay, gradient_accumulation_steps,
               training_batch_size, eval_batch_size, bidirectional, dropout, num_layers, hidden_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = f"{output_dir}/lstm-finetuned-{dataset['name']}"

    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_data, test_data, label_dict = prepare_data(dataset, tokenizer, Dataset2)
    vocab_size = tokenizer.vocab_size
    embedding = None

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              collate_fn=collate_for_lstm,
                              batch_size=training_batch_size)

    model = LSTM(vocab_size,
                 len(label_dict),
                 bidirectional=bidirectional,
                 dropout=dropout,
                 num_layers=num_layers,
                 hidden_size=hidden_size
                 ).to(device)


    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    t_total = len(train_loader) // gradient_accumulation_steps * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    logging.info(f"Training model on {device}")
    global_step = 0
    training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs, desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            flat_docs, offsets, labels = batch
            outputs = model(flat_docs, offsets, labels)

            loss = outputs[0]
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            training_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            logging_steps = len(train_data) // 16
            if global_step % logging_steps == 0:
                acc, eval_loss = evaluate_lstm(model, test_data, eval_batch_size, device, output_dir)
                logging.info(f"Epoch {epoch} Step {step} Loss {loss.item()} Eval loss {eval_loss} Acc {acc}")

    # eval
    acc, eval_loss = evaluate_lstm(model, test_data, eval_batch_size, device, output_dir)
    logging.info(f"Evaluation loss: {eval_loss}")
    logging.info(f"Evaluation accuracy: {acc}")
    

def evaluate_ml(trainer, test_data, output_dir):
    # accuracy
    test_text = [test_data.text['input_ids'][i] for i in range(len(test_data))]
    test_labels = [test_data.labels[i] for i in range(len(test_data))]

    y_preds = trainer.predict(test_text)
    y_true = test_labels
    acc = accuracy_score(y_true, y_preds)
    logging.info(f"Test accuracy: {acc}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_preds)
    logging.info(f"Confusion matrix:\n{cm}")

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save results to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump({"acc": acc}, f)

def evaluate_transformer(trainer, test_data, output_dir):
    # Existing code for Transformer models
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

def evaluate_lstm(model, test_data, eval_batch_size, device, output_dir):
    data_loader = DataLoader(test_data,
                             shuffle=False,
                             collate_fn=collate_for_lstm,
                             batch_size=eval_batch_size)
    all_logits = []
    all_targets = []
    eval_steps, eval_loss = 0, 0.0
    for batch in tqdm(data_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            flat_inputs, lengths, labels = batch
            outputs = model(flat_inputs, lengths, labels)
            all_targets.append(labels.detach().cpu())

        eval_steps += 1
        loss, logits = outputs[:2]
        eval_loss += loss.mean().item()
        all_logits.append(logits.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    eval_loss /= eval_steps
    preds = np.argmax(logits, axis=1)
    acc = (preds == targets).sum() / targets.size
    
    # confusion matrix
    cm = confusion_matrix(targets, preds)
    logging.info(f"Confusion matrix:\n{cm}")

    # create file if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # append result to file
    with open(f"{output_dir}/eval_results.json", "a") as f:
        f.write("\n")
        json.dump({"acc": acc, "model parameter": str(model)}, f)

    return acc, eval_loss


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

    #LSTM args
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers.')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size.')

    args = parser.parse_args()


    config = vars(args)

    logging.info("Starting...")
    logging.debug("Arguments: %s", args)

    # Start training
    logging.info(f"Loading {args.dataset} data...")
    dataset = load_data(args.dataset)
    
    if args.model == "LSTM":
        train_lstm(dataset,
                    config["output_dir"],
                    epochs=config["num_train_epochs"],
                    warmup_steps=config["warmup_steps"],
                    learning_rate=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    training_batch_size=config["batch_size"],
                    eval_batch_size=config["batch_size"],
                    bidirectional=config["bidirectional"],
                    num_layers=config["num_layers"],
                    hidden_size=config["hidden_size"],
                    dropout=config["dropout"],
                    gradient_accumulation_steps=config["gradient_accumulation_steps"]
                    )

    elif args.model in ("SVM","RandomForest","LogisticRegression","NaiveBayes"):
        train_ML(dataset, args.model, config["output_dir"])

    else:
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
    
# ut python main.py FakeTrue SVM --output_dir output #run SVM in CLI