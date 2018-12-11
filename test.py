# Imports
import pandas as pd
import numpy as np
import pickle
from gensim.models import KeyedVectors
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from torch import nn
import time
import math

class QuestionPairLSTM(torch.nn.Module):

    def __init__(self, embedding, hidden, num_layers, batch_size, device="cpu"):
        super(QuestionPairLSTM, self).__init__()

        # load embedding matrix
        emb_matrix = pickle.load(open("data/embedding_matrix.p", "rb"))

        embeddings = torch.tensor(emb_matrix, dtype=torch.float).to(device)
        self.embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embedding.weight = torch.nn.Parameter(embeddings)
        self.embedding.require_grad = False

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden = hidden

        self.device = device
        self.lstm = torch.nn.LSTM(embedding, hidden, num_layers, batch_first=True)

        # self.hidden = self.zero_hidden()

    
    def forward(self, q):
        y = self.zero_hidden(q.shape[0])
        return self.lstm(q, y)

    def zero_hidden(self, leng):
        return (torch.zeros(self.num_layers, leng, self.hidden).to(self.device),
                torch.zeros(self.num_layers, leng, self.hidden).to(self.device))

class LoadQuestions(Dataset):
    def __init__(self, tok, size):
        self.tok = tok
        self.size = size
    
    def __len__(self):
        # return 3
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
        
        return (s['id'], torch.tensor(q1), torch.tensor(q2))

def exponent_neg_manhattan_distance(left, right):
    return torch.exp(-torch.sum(torch.abs(left-right), dim=1))

if __name__ == "__main__":

    # params
    embedding_size = 300
    hidden_dimension = 50
    layers = 1
    epochs = 10
    sentence_size = 120

    # load vocab and convert to list
    vocab = pickle.load(open("data/vocab_test.p", "rb"))

    # generate inverse vocab
    inf_vocab = {x : y for y, x in enumerate(vocab)} 

    # size of vocab
    V = len(vocab)
    batch_size = 128

    tokenized = pd.read_csv("data/tokenized_questions_test.csv")

    emb_matrix = pickle.load(open("data/embedding_matrix_test.p", "rb"))
    embeddings = torch.tensor(emb_matrix, dtype=torch.float)
    embedding = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
    embedding.weight = torch.nn.Parameter(embeddings)

    # Epochs
    for ep in range(9,10):
        
        model = QuestionPairLSTM(embedding_size, hidden_dimension, layers, batch_size, "cuda")
        model.cuda()
        model.load_state_dict(torch.load('data/lstm_model_epoch_adadelta_' + str(ep + 1) + '.pt'))
        model.eval()

        # # Show epoch number
        # print("Running epoch " + str(ep))
        
        # Load question pairs
        questions = LoadQuestions(tokenized, sentence_size)

        ids = []
        outputs = []
        
        # Calculate number of batches
        num_batches = int(math.ceil(len(questions) / batch_size))

        with torch.no_grad():

            for x, (idlist, q1, q2) in enumerate(DataLoader(questions, batch_size, shuffle = False, drop_last = False)):
            # for i, q in enumerate(questions):

                # print('\rEpoch {:d}, question {:d}'.format(ep, i), end='', flush=True)

                num = x + 1
                # How far are we?
                perc = (float(num) / num_batches) * 100.0

                # Output
                print('\rProcess: batch {:d} of {:d} ({:.3f}%)'.format(num, num_batches, perc), end='', flush=True)

                # print(ids.numpy())

                for x in idlist.numpy():
                    ids.append(x)

                output_q1, hidden_1 = model(embedding(q1))
                output_q2, hidden_2 = model(embedding(q2))
                
                scores = exponent_neg_manhattan_distance(hidden_1[0].view(len(idlist), -1),
                                                         hidden_2[0].view(len(idlist), -1))

                for x in scores.numpy():
                    outputs.append(int(round(x)))

                # print(scores)

                # dataframe = pd.DataFrame({
                #     'test_id': ids,
                #     'is_duplicate'
                # })


            # print(r)
            # if i > 15:
            #     break
        dataframe = pd.DataFrame({
            'test_id': ids,
            'is_duplicate': outputs
        })

        # print(dataframe)

        print()
        print("Storing dataframe...")

        dataframe.to_csv('data/output_epoch_adadelta_{:d}.csv'.format(ep), index=False, columns=['test_id', 'is_duplicate'])