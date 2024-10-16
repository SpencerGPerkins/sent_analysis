import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set device to CUDA or CPU
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("No GPU, using CPU")
    
    
class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5):
        super(SentimentRNN, self).__init__()
        
        self.output_dim = output_dim  # Set output dimension to 3 (for 3 classes)
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)
        # Fully connected layer (for multi-class classification)
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        # Get word embeddings
        embeds = self.embedding(x)  # shape: (B, S, embedding_dim)
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embeds, hidden)  # hidden state is passed here
        # Get the output of the last time step (shape: [batch_size, hidden_dim])
        lstm_out = lstm_out[:, -1, :]  # Use the last time step output
        out = self.batch_norm(lstm_out)
        out = self.dropout(out)  
        # Fully connected layer (shape: [batch_size, output_dim])
        out = self.fc(out)
        
        return out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state for LSTM"""
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        
        return hidden



def preprocess_text(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s   

def one_hot_encode(labels):
    encoded_labels = []
    for label in labels:
        if label.lower() == 'positive':
            encoded_labels.append([1, 0, 0])
        elif label.lower() == 'neutral':
            encoded_labels.append([0,1,0])
        elif label.lower() == 'negative':
            encoded_labels.append([0,0,1])
        else:
            raise ValueError('Unidentified sentiment found.')
    
    return encoded_labels

def sent_encode(labels):
    # Mapping from string labels to integer indices
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    # Convert string labels to integer indices
    numeric_labels = torch.tensor([label_map[label] for label in labels])
    
    return numeric_labels

def tokens(x_train, y_train, x_val, y_val):
    """ Generate Tokens and One-hot encoding"""
    word_list = []
    
    stop_words = set(stopwords.words('english'))
    for text in x_train:
        for word in text.lower().split():
            word = preprocess_text(word)
            if word not in stop_words and word != '':
                word_list.append(word)
                
    corpus = Counter(word_list)
    # Sort on basis of most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # Dict
    onehot_dict = {w:i+1 for i, w in enumerate(corpus_)}
    
    # Tokenize
    final_list_train, final_list_test = [],[]
    for text in x_train:
        final_list_train.append([onehot_dict[preprocess_text(word)] for word in text.lower().split()
                                 if preprocess_text(word) in onehot_dict.keys()])
    for text in x_val:
        final_list_test.append([onehot_dict[preprocess_text(word)] for word in text.lower().split()
                                 if preprocess_text(word) in onehot_dict.keys()])
        
    # Pad sequences using PyTorch
    padded_train = pad_sequence([torch.tensor(seq) for seq in final_list_train], batch_first=True, padding_value=0)
    padded_test = pad_sequence([torch.tensor(seq) for seq in final_list_test], batch_first=True, padding_value=0)
    
    # # One-hot encoding labels   
    # encoded_train = one_hot_encode(y_train)
    # encoded_test = one_hot_encode(y_val)
    # Numeric labels for CrossEntropy loss
    encoded_train = sent_encode(y_train)
    encoded_test = sent_encode(y_val)
    
    # Convert encoded labels to torch tensors
    encoded_train = torch.tensor(encoded_train, dtype=torch.float32)
    encoded_test = torch.tensor(encoded_test, dtype=torch.float32)
    
    return padded_train, encoded_train, padded_test, encoded_test, onehot_dict

def sum_correct(pred, label):
    # Get the predicted classes by taking the class with the highest probability
    _, pred_classes = torch.max(pred, 1)
    
    # Compare the predicted classes with the true labels
    correct_predictions = torch.sum(pred_classes == label).item()
    
    return correct_predictions


def main():
    
    # Load dataset and define X, y
    df = pd.read_csv('../data/social_media/sentiment_analysis.csv')
    X, y = df['text'].values, df['sentiment'].values
    # Training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    print(f'Shape of train data is {X_train.shape}.')
    print(f'Shape of test data is {X_test.shape}.')
    print(type(X_train[0]))
    
    X_train, y_train, X_test, y_test, vocab = tokens(X_train, y_train, X_test, y_test)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_test, y_test)
    
    tr_batch_size = 60
    vl_batch_size = 60
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=tr_batch_size)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=vl_batch_size)
    
    dataiter = iter(train_loader)
    sample_x, sample_y = next(dataiter)
    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print('Sample input: \n', sample_y)
    
    no_layers = 2
    vocab_size = len(vocab) + 1 
    embedding_dim = 64
    output_dim = 3
    hidden_dim = 128
    model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5)
    model.to(device)
    print(model)
    
    # Loss and optimization
    lr=0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    clip = 5
    epochs = 500
    valid_loss_min = np.inf
    # train for some number of epochs
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(epochs):
        train_losses = []
        train_corr = 0.0
        model.train()
        
        # initialize hidden state for the training set using the dynamic batch size
        for inputs, labels in train_loader:
            batch_size = inputs.size(0)  # Capture the batch size for this batch dynamically
            h = model.init_hidden(batch_size)  # Use the dynamic batch size for hidden state initialization
            
            inputs, labels = inputs.to(device), labels.to(device)   
            h = tuple([each.data for each in h])  # Detach hidden states from history
            
            model.zero_grad()
            output, h = model(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, labels.long())
            loss.backward()
            train_losses.append(loss.item())
            
            # calculating accuracy
            total_correct = sum_correct(output, labels)
            train_corr += total_correct
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
    
        val_losses = []
        val_corr = 0.0
        model.eval()

        # initialize hidden state for the validation set using the dynamic batch size
        for inputs, labels in val_loader:
            batch_size = inputs.size(0)  # Capture the batch size for this batch dynamically
            val_h = model.init_hidden(batch_size)  # Use the dynamic batch size for hidden state initialization
            
            val_h = tuple([each.data for each in val_h])  # Detach hidden states from history
            inputs, labels = inputs.to(device), labels.to(device)

            output, val_h = model(inputs, val_h)
            val_loss = criterion(output, labels.long())

            val_losses.append(val_loss.item())
            total_correct = sum_correct(output, labels)
            val_corr += total_correct
                
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_corr / len(train_loader.dataset)
        epoch_val_acc = val_corr / len(val_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
        print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
        
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), '../models/lstm/state_dict.pt')
            print(f'Validation loss decreased ({valid_loss_min:.6f} --> {epoch_val_loss:.6f}).  Saving model ...')
            valid_loss_min = epoch_val_loss
            
        print(25 * '==')
    fig = plt.figure(figsize = (20, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_tr_acc, label='Train Acc')
    plt.plot(epoch_vl_acc, label='Validation Acc')
    plt.title("Accuracy")
    plt.legend()
    plt.grid()
        
    plt.subplot(1, 2, 2)
    plt.plot(epoch_tr_loss, label='Train loss')
    plt.plot(epoch_vl_loss, label='Validation loss')
    plt.title("Loss")
    plt.legend()
    plt.grid()

    plt.show()
    
        
if __name__ == "__main__":
    main()
    