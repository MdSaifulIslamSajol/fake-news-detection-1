import csv
import torch
import re
import string
from sklearn.model_selection import train_test_split

def prepare_data(dataset, tokenizer, dataset_class):
    
    train_text, train_labels = dataset['train']
    test_text, test_labels = dataset['test']
    
    if tokenizer is not None:
        # Tokenize texts
        encoded_train_inputs = tokenizer(train_text, truncation=True, padding=True)
        encoded_test_inputs = tokenizer(test_text, truncation=True, padding=True)
    else:
        # If tokenizer is None (e.g. for SVM), use the raw text
        encoded_train_inputs = {"input_ids": train_text}
        encoded_test_inputs = {"input_ids": test_text}

    # Encode labels
    label_dict = dataset['label_dict']
    encoded_train_labels = [label_dict[label] for label in train_labels]
    encoded_test_labels = [label_dict[label] for label in test_labels]
    # create dataset
    train_data = dataset_class(encoded_train_inputs, encoded_train_labels)
    test_data = dataset_class(encoded_test_inputs, encoded_test_labels)

    return train_data, test_data, dataset['label_dict']



def load_data(key):

    dataset = {"name": key}

    if key == 'LIAR':
        load_LIAR(dataset)
    elif key == 'FakeTrue':
        load_FakeTrue(dataset)

    else:
        raise ValueError(f"Unknown dataset: {key}")

    return dataset

def prep_FakeTrue():
    with open('data/FakeTrue/Fake.csv', 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        fakes = ['fake '+row[1] for row in csvreader]
    
    with open('data/FakeTrue/True.csv', 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        trues = ['true '+row[1] for row in csvreader]

    all_texts = trues+fakes
    all_texts = [clean_text(text) for text in all_texts]
    train, test = train_test_split(all_texts, test_size=0.25, shuffle=True, random_state = 0)

    f1 = open('data/FakeTrue/train.txt','w')
    for item in train:
        f1.write(item+'\n')
    f1.close()

    f2 = open('data/FakeTrue/test.txt','w')
    for item in test:
        f2.write(item+'\n')
    f2.close()

def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?:\/\/\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_FakeTrue(dataset):
    # load training data
    with open("data/FakeTrue/train.txt", "r", encoding="utf-8") as f:
        raw_train = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_train]
    # first element is the label
    train_text = [" ".join(line[1:]) for line in list_of_words]
    train_labels = [line[0] for line in list_of_words]
    # load test data
    with open("data/FakeTrue/test.txt", "r", encoding="utf-8") as f:
        raw_test = [line.strip() for line in f]
    list_of_words = [line.split() for line in raw_test]
    # first element is the label
    test_text = [" ".join(line[1:]) for line in list_of_words]
    test_labels = [line[0] for line in list_of_words]
    # create label dictionary
    label_dict = create_dict(set.union(set(train_labels), set(test_labels)))
    # add to dataset
    dataset["train"] = (train_text, train_labels)
    dataset["test"] = (test_text, test_labels)
    dataset["label_dict"] = label_dict


def create_dict(labels):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    return label_dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.text.items()}
        item['labels'] = self.labels[index]
        return item

    def __len__(self):
        return len(self.labels)

class Dataset2(torch.utils.data.Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __getitem__(self, index):
        text = self.text[index]
        label = self.labels[index]
        return text, label

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    prep_FakeTrue()