# Imports
import pandas as pd
import numpy as np
import pickle
from gensim.models import KeyedVectors
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from torch import nn

class QuestionPairLSTM(torch.nn.Module):

    def __init__(self, embedding, hidden, num_layers, batch_size, device="cpu"):
        super(QuestionPairLSTM, self).__init__()

        # load embedding matrix
        emb_matrix = pickle.load(open("data/embedding_matrix.p", "rb"))

        embeddings = torch.tensor(emb_matrix, dtype=torch.float).to(device)
        self.embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embedding.weight = torch.nn.Parameter(embeddings)
        # self.embedding.require_grad = False

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden = hidden

        self.device = device
        self.lstm = torch.nn.LSTM(embedding, hidden, num_layers, batch_first=True)

        # self.hidden = self.zero_hidden()

    
    def forward(self, q):
        x = self.embedding(q.to(self.device))
        y = self.zero_hidden()
        return self.lstm(x, y)

        # etc etc

    def zero_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden).to(self.device))

class LoadQuestions(Dataset):
    def __init__(self, tok, size):
        self.tok = tok
        self.size = size
    
    def __len__(self):
        return len(self.tok)
    
    def __getitem__(self, idx):
        s = self.tok.iloc[idx]
        
        # Create empty paddings
        q1 = [0] * self.size
        q2 = [0] * self.size
        
        # Fetch raw questions (literal_eval because I fucked up the formatting in PD)
        q1_raw = ast.literal_eval(s['question1'])
        q2_raw = ast.literal_eval(s['question2'])
        
        # Add raw questions to the end of the padding
        q1[(self.size - len(q1_raw)):self.size] = q1_raw
        q2[(self.size - len(q2_raw)):self.size] = q2_raw
        
        return (torch.tensor(q1), torch.tensor(q2), torch.tensor(float(s['is_duplicate'])))

def exponent_neg_manhattan_distance(left, right):
    return torch.exp(-torch.sum(torch.abs(left-right), dim=1))

if __name__ == "__main__":

    # params
    embedding_size = 300
    hidden_dimension = 30
    layers = 1
    batch_size = 32
    epochs = 10
    sentence_size = 120

    # load vocab and convert to list
    vocab = pickle.load(open("data/vocab_simple.p", "rb"))

    # generate inverse vocab
    inf_vocab = {x : y for y, x in enumerate(vocab)} 

    # size of vocab
    V = len(vocab)

    tokenized = pd.read_csv("data/tokenized_questions.csv")

    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.SmoothL1Loss()

    model = QuestionPairLSTM(embedding_size, hidden_dimension, layers, batch_size, "cuda")
    model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)

    # Epochs
    for ep in range(1, (epochs + 1)):
        
        # Show epoch number
        print("Running epoch " + str(ep))
        
        questions = LoadQuestions(tokenized, sentence_size)
        
        num_batches = int(len(questions) // batch_size)

        for x, (q1, q2, wanted) in enumerate(DataLoader(questions, batch_size, shuffle = True, drop_last = True)):
            
            # How far are we?
            perc = float(x) // num_batches * 100.0

            # Output
            print('\rProcess: batch {:d} of {:d} ({:.3f}%)'.format(x, num_batches, perc), end='', flush=True)

            # Get output for first questions
            output_q1, hidden_1 = model(q1)
            output_q2, hidden_2 = model(q2)
            
            scores = exponent_neg_manhattan_distance(hidden_1[0].view(batch_size, -1),
                                                     hidden_2[0].view(batch_size, -1))

            wanted = wanted.to(model.device)

            # Calculate loss
            loss = loss_fn(scores, wanted)
                
            # Backpropagate
            loss.backward()

            # Update weights
            optimizer.step()
                
        # Show loss
        print('Loss: {:6.4f}'.format(loss.item()))

