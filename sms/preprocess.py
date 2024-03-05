

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
import pickle

# Sample data loading function
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    labels, texts = [], []
    for line in lines:
        label, text = line.split('\t', 1)
        labels.append(label)
        texts.append(text.strip())
    return labels, texts


# Split data into train, test, and validation sets
def split_data(labels, texts, test_size=0.15, val_size=0.15):
    texts_train, texts_temp, labels_train, labels_temp = train_test_split(texts, labels, test_size=test_size + val_size)
    val_size_adjusted = val_size / (test_size + val_size)  # Adjust val size for remaining data
    texts_val, texts_test, labels_val, labels_test = train_test_split(texts_temp, labels_temp, test_size=val_size_adjusted)
    return texts_train, texts_val, texts_test, labels_train, labels_val, labels_test


# Tokenization and vocab building
def tokenize_and_build_vocab(texts_train):
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, texts_train), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab


# Function to save tokenized data and labels
def save_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


# Function to save the tokenizer and vocabulary
def save_tokenizer_and_vocab(file_path, tokenizer, vocab):
    with open(file_path, 'wb') as file:
        pickle.dump({'tokenizer': tokenizer, 'vocab': vocab}, file)



# Example of tokenizing a single dataset
def tokenize_data(texts, tokenizer, vocab):
    tokenized_texts = [torch.tensor(vocab(tokenizer(text)), dtype=torch.long) for text in texts]
    return tokenized_texts

# Assuming your file path is 'data.txt'
filepath = 'SMSSpamCollection'
labels, texts = load_data(filepath)
texts_train, texts_val, texts_test, labels_train, labels_val, labels_test = split_data(labels, texts)
tokenizer, vocab = tokenize_and_build_vocab(texts_train)
tokenized_texts_train = tokenize_data(texts_train, tokenizer, vocab)
tokenized_texts_val = tokenize_data(texts_val, tokenizer, vocab)
tokenized_texts_test = tokenize_data(texts_test, tokenizer, vocab)

# Now you have tokenized_texts_train, tokenized_texts_val, and tokenized_texts_test ready for your model
import pickle


# Save the tokenized texts and labels
save_data('tokenized_texts_train.pkl', tokenized_texts_train)
save_data('labels_train.pkl', labels_train)
save_data('tokenized_texts_val.pkl', tokenized_texts_val)
save_data('labels_val.pkl', labels_val)
save_data('tokenized_texts_test.pkl', tokenized_texts_test)
save_data('labels_test.pkl', labels_test)

# Save the tokenizer and vocabulary for future use
save_tokenizer_and_vocab('tokenizer_vocab.pkl', tokenizer, vocab)
