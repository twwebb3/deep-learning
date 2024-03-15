
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Function to load data
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# Function to load the tokenizer and vocabulary
def load_tokenizer_and_vocab(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['tokenizer'], data['vocab']


# Load the tokenized texts and labels
tokenized_texts_train = load_data('tokenized_texts_train.pkl')
labels_train = load_data('labels_train.pkl')
tokenized_texts_val = load_data('tokenized_texts_val.pkl')
labels_val = load_data('labels_val.pkl')
tokenized_texts_test = load_data('tokenized_texts_test.pkl')
labels_test = load_data('labels_test.pkl')

# Load the tokenizer and vocabulary
tokenizer, vocab = load_tokenizer_and_vocab('tokenizer_vocab.pkl')

# First, we need to encode the labels as integers
label_encoder = LabelEncoder()
labels_train_encoded = label_encoder.fit_transform(labels_train)
labels_val_encoded = label_encoder.transform(labels_val)


# Since RNNs in PyTorch require inputs to have the same length, we pad the sequences
def pad_sequences(sequences, max_len):
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(max_len, len(seq))
        padded_sequences[i, :length] = seq[:length]
    return padded_sequences


# Determine the maximum sequence length
max_len = max(max(len(seq) for seq in tokenized_texts_train), max(len(seq) for seq in tokenized_texts_val))

# Pad the tokenized texts
padded_texts_train = pad_sequences(tokenized_texts_train, max_len)
padded_texts_val = pad_sequences(tokenized_texts_val, max_len)

# Convert labels to tensors
labels_train_tensor = torch.tensor(labels_train_encoded, dtype=torch.long)
labels_val_tensor = torch.tensor(labels_val_encoded, dtype=torch.long)

# Create DataLoader instances for training and validation sets
batch_size = 64
train_dataset = TensorDataset(padded_texts_train, labels_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(padded_texts_val, labels_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the RNN model (84.09 % accuracy)
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=1, nonlinearity='relu', batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)  # Apply dropout
        out = self.fc(last_output)
        return out


# Instantiate the model
vocab_size = len(vocab)  # Vocabulary size
embedding_dim = 100  # Size of the embedding vectors
hidden_dim = 256  # Number of features in the hidden state
output_dim = len(label_encoder.classes_)  # Number of output classes

model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training the model
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Total Training Loss: {total_loss}')

    # Validation
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Accuracy: {correct / total * 100:.2f}%')
