Fake News Detection Project

Dependencies

The following libraries are required to run the project:

transformers
torch
scikit-learn
Usage

To run the project, use the following command:

python main.py [DATASET] [MODEL] [OPTIONS]

DATASET can be:
FakeTrue
MODEL can be:
SVM
LSTM
RandomForest
NaiveBayes
LogisticRegression
BERT
DISTILBERT
ALBERT
OPTIONS can be:
--learning_rate
--batch_size
--num_train_epochs
--dropout
--num_layers
--hidden_size
--...
Examples

Transformers:
python main.py FakeTrue BERT --learning_rate=1e-5 --batch_size=16 --num_train_epochs=2 --dropout=0

LSTM:
python main.py FakeTrue LSTM --learning_rate=0.002 --batch_size=32 --num_train_epochs=30 --dropout=0.5 --hidden_size=128 --num_layers=2

Traditional ML:
python main.py FakeTrue SVM