"""Read the paper "Convolutional Neural Networks for Sentence Classification", 
and implement a CNN-based neural network for sentence classification (Chinese). 
The datasets are already processed in the current directory (each line is a datapoint with text + label),
and you need to construct the vocabulary by yourself 
(you may need jieba to tokenize sentences and construct a vocabulary)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import jieba
import numpy as np
import random
import re
from torch.utils.data import (TensorDataset, DataLoader, RandomSampler)

# Load text data
def load_text(path):
    """
    Load text data from a file. Each line is a datapoint with Chinese text + label.

    Returns:
        texts(List(str))
        labels(List(int))
    
    """

    texts = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            text, label = line.split('\t')
            texts.append(text)
            labels.append(int(label))
    return texts, labels

# Data preparation
# Tokenize sentences and construct vocabulary
def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.
    
    Args:
        texts (List[str]): List of text data
    
    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """
    max_len = 0
    word2idx = {}
    tokenized_texts = []

    word2idx["<填充>"] = 0
    word2idx["<未知>"] = 1
    idx = 2

    for sent in texts:
        sent = re.sub(r'[^\u4e00-\u9fa5]+', '', sent)
        tokenized_sent = jieba.lcut(sent)
        tokenized_texts.append(tokenized_sent)

        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len

def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_idxs = []
    
    for tokenized_sent in tokenized_texts:
        tokenized_sent += ["<填充>"] * (max_len - len(tokenized_sent))
        input_idx = [word2idx.get(token, word2idx["<未知>"]) for token in tokenized_sent]
        input_idxs.append(input_idx)

    return np.array(input_idxs)

#Create Pytorch DataLoader
def data_loader(inputs, labels, batch_size=50):
    """
    Convert  datasets to torch.Tensors and load them to
    DataLoader.

    """

    inputs, labels = tuple(torch.tensor(data) for data in [inputs, labels])

    batch_size = 50

    data = TensorDataset(inputs, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, batch_size, sampler=sampler)


    return dataloader

#Create CNN Model
class TextCNN(nn.Module):

    def __init__(self, vocab_size=None, embedding_dim=300,
              filter_sizes=[3,4,5], filter_nums=[100,100,100],
              classes_num=4, drop_out=0.5):
        super(TextCNN, self).__init__()

        #Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, 0, 5.0)

        #Conv
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 
                      filter_nums[i], 
                      filter_sizes[i])
            for i in range(len(filter_sizes))
        ])

        #FC and dropout
        self.fc = nn.Linear(np.sum(filter_nums), classes_num)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, input_idxs):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """
        #embedding  size(batch_size, max_sent_length, embedding_dim)
        x = self.embedding(input_idxs).float()
        #size(batch_size, embedding_dim, max_sent_length)
        x_permu = x.permute(0, 2, 1)
        #convolution  size(batch_size, n_filter, length_after_conv)
        x_conv_list = [F.relu(conv(x_permu)) for conv in self.convs]
        #max_pool  size(batch_size, n_filter, 1)
        x_pooling_list = [F.relu(nn.MaxPool1d(x_conv, kernel_size=x_conv.shape[2])) for x_conv in x_conv_list]
        #cat size(batch_size, 3*n_filter)
        x_cat = torch.cat([x_pooling.squeeze(2) for x_pooling in x_pooling_list], dim=1)
        #dropout 
        x_dropout = self.drop_out(x_cat)
        #fc  size(batch_size, n_classes)
        logits = self.fc(x_dropout)

        return logits
#Training
def train(model, optimizer, loss_fn, device, train_dataloader, val_dataloader=None, epochs=20):
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    #training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        #train on batch sample
        for step, batch in enumerate(train_dataloader):
            batch_input_idxs, batch_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(batch_input_idxs)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss/len(train_dataloader)
        # validation
        if val_dataloader is not None:
            val_loss, val_accuracy = evaluate(model, val_dataloader, device, loss_fn)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_improvement = 0
                torch.save(model.state_dict(), 'best_model.pt') # save the best model parameters
            else:
                epochs_since_improvement += 1
            # early stopping
            if epochs_since_improvement > 3:
                break

            print(f"{'Epoch: ':^7}{epoch+1:^7} | {'Train Loss: ':^12}{avg_loss:^12.6f} | {'Val Loss: ':^10}{val_loss:^10.6f} | {'Val Acc: ':^9}{val_accuracy:^9.2f}")
            
    print("\n")
    print(f"Training complete! Lowest validation loss:{best_val_loss:.2f}%.")
#Evaluate
def evaluate(model, dataloader, device, loss_fn):
    model.eval()

    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_input_idxs, batch_labels = tuple(t.to(device) for t in batch)

            logits = model(batch_input_idxs)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()

            predict_labels = torch.argmax(logits, dim=1).flatten().to(batch_labels.device) # move predicted labels to the same device as true labels
            accuracy = (predict_labels == batch_labels).float().mean().item() * 100
            total_accuracy += accuracy

            total_samples += batch_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    return avg_loss, avg_accuracy

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":
    # Use GPU for Training
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    #Load texts
    train_texts, train_labels = load_text("train.txt")
    val_texts, val_labels = load_text("dev.txt")
    total_texts = np.array(train_texts + val_texts)
    total_labels = np.array(train_labels + val_labels)

    total_tokenized_texts, word2idx, max_len = tokenize(total_texts)
    total_input_idxs = encode(total_tokenized_texts, word2idx, max_len)

    #split the orignal train and validation sets
    train_input_idxs = total_input_idxs[:len(train_texts)]
    train_labels = total_labels[:len(train_texts)]
    val_input_idxs = total_input_idxs[len(train_texts):]
    val_labels = total_labels[len(train_texts):]

    #load data
    train_dataloader = data_loader(train_input_idxs, train_labels)
    val_dataloader = data_loader(val_input_idxs, val_labels)

    #set seed
    set_seed(42)

    #create model
    model = TextCNN(vocab_size=len(word2idx), embedding_dim=300, filter_sizes=[3,4,5], filter_nums=[100,100,100], classes_num=4, drop_out=0.5)
    model.to(device)

    #create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #create loss function
    loss_fn = nn.CrossEntropyLoss()

    #train
    train(model, optimizer, loss_fn, device, train_dataloader, val_dataloader, epochs=40)

    #test
    test_texts, test_labels = load_text("test.txt")
    test_tokenized_texts, _, _ = tokenize(test_texts)
    test_input_idxs = encode(test_tokenized_texts, word2idx, max_len)
    test_dataloader = data_loader(test_input_idxs, test_labels)
    test_loss, test_accuracy = evaluate(model, test_dataloader, device, loss_fn)
    print(f"Test Loss: {test_loss:^10.6f} | Test Acc: {test_accuracy:^9.2f}")












