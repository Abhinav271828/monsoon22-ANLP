# %%
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import math
from data2 import *

# %%
class RNNCell(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.hs = hidden_size
        self.is_ = input_size

        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u = nn.Linear(input_size, hidden_size, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, hidden_state, input):
        #             [hs]          [is]

        wh = self.w(hidden_state)
        # [hs]
        if (math.isnan(wh[0][0].item())):
            print("weights", self.w.weight)
            exit()

        ux = self.u(input)
        # [hs]

        next_hidden = self.tanh(wh + ux)
        # [hs]

        return next_hidden

# %%
class RNN(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers = 1):
        super().__init__()
        self.cell = RNNCell(hidden_size, input_size)
        self.n = num_layers
    
    def forward(self, init_hidden_state, inputs):
        #             [hs]               [ml, is]
        
        for _ in range(self.n):
            hidden_states = []
            hidden_state = init_hidden_state
            for input in inputs:
                # [is]
                hidden_state = self.cell(hidden_state, input)
                hidden_states.append(hidden_state)
                # [bz, hs]
            
            hidden_states = torch.stack(hidden_states, dim=1)

            inputs = hidden_states

        return hidden_states

class LanguageModel(nn.Module):
    def __init__(self, rnn, vocab_size, emb_matrix):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 300).from_pretrained(emb_matrix)
        self.rnn = rnn
        self.linear = nn.Linear(rnn.cell.hs, vocab_size)
    
    def forward(self, ihs, batch):
    #                 [hs] [ml]
        batch = self.emb(batch)
        # [ml, 300]
        hs = self.rnn(ihs, batch)
        # [ml, hs]

        final_state = hs[:,-1,:] # [bz, hs]
        logits = self.linear(final_state)

        return logits

    def train_epoch(self, ds, optimiser, loss_fn):
        for sentence in tqdm(ds):
            optimiser.zero_grad()
            contexts, words = sentence
            logits = self.forward(torch.zeros(contexts.shape[0]).to(device), contexts)

            loss = loss_fn(logits, words)
            loss.backward()

            optimiser.step()
   

    def train(self, num_epochs, lr = 1e-5):
        optimiser = torch.optim.SGD(self.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss()
        #train_dl = DataLoader(train_ds, batch_size = 128)
        #dev_dl = DataLoader(dev_ds, batch_size = 128)

        prev_perp = math.inf
        for _ in range(num_epochs):
            self.train_epoch(train_ds, optimiser, loss_fn)
            loss = self.get_loss(dev_ds, loss_fn)
            print("Loss on validation set:", loss)
            perp = self.get_perp(dev_ds)
            print("Perp on validation set:", perp)

            if (perp < prev_perp):
                torch.save(self, 'lm2.pth')
            else: break

    def get_loss(self, dl, loss_fn):
        total = 0
        for batch in dl:
            contexts, words = batch
            pred = self.forward(torch.zeros(contexts.shape[0], self.rnn.cell.hs).to(device), contexts)
            loss = loss_fn(pred, words)
            total += loss.item()

        return total

    def get_perp(self, dl):
        loss = nn.CrossEntropyLoss()
        sum_perp = 0
        count = 0
        for batch in dl:
            contexts, words = batch
            pred = self.forward(torch.zeros(contexts.shape[0], self.rnn.cell.hs).to(device), contexts)

            cel = loss(pred, words).item()
            perp = math.exp(cel)

            sum_perp += perp
            count += len(batch)

        return (sum_perp/count)


rnn = RNN(300, 300)
lm = LanguageModel(rnn, len(train_ds.vocab), train_ds.embeddings).to(device)
lm.train(15)

def perp_file(lm, inp_file, out_file, ds):
    lf = nn.CrossEntropyLoss()
    with open(inp_file, 'r') as f:
        with open(out_file, 'w') as g:
            for line in f:
                words = [word.lower() for word in word_tokenize(line)]
                indices = [ds.words2indices[word] if word in ds.vocab \
                           else (len(ds.vocab)-1)
                                for word in words]
                embeddings = [ds.embeddings[i] for i in indices]

                contexts = []
                words = []
                for i in range(len(embeddings) - ds.context_size):
                    contexts.append(torch.stack(embeddings[i:i + ds.context_size]))
                    words.append(indices[i + ds.context_size])
        
                contexts = torch.stack(contexts)
                words = torch.tensor(words)

                outputs = lm(torch.zeros(len(words), 300).to(device), contexts)
                loss = lf(outputs, words).item()
                perp = math.exp(loss)

                g.write(line[:-1] + '\t' + str(perp) + '\n')

def hyper():
    lrs = [0.001, 0.0005, 0.0001]
    ns = [1, 2, 3]

    test_dl = DataLoader(test_ds, batch_size = 128)
    with open('hyper_ffn.txt', 'w') as f:
        for lr in lrs:
            for n in ns:
                lm = LanguageModel(RNN(300, 300), len(train_ds.vocab))
                print(f"lr={lr}-n={n}")
                lm.train(15, lr)

                p = lm.get_perp(test_dl)

                f.write(f"lr={lr}-n={n}: {p}\n")
