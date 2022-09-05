from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torch
import math
from data import *

# %% [markdown]
# The feedforward model

# %%
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, context_size = 4,
                       hidden_size_1 = 300, hidden_size_2 = 300):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features = 300 * context_size,
                                           out_features = hidden_size_1),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features = hidden_size_1,
                                           out_features = hidden_size_2),
                                 nn.ReLU())
        self.output = nn.Sequential(nn.Linear(in_features = hidden_size_2,
                                           out_features = vocab_size),
                                    nn.ReLU())

        self.layer = nn.Sequential(self.fc1, self.fc2, self.output)
        #self.softmax = nn.Softmax()
    
    def forward(self, batch_ctx):
        batch_ctx = batch_ctx.flatten(1,2)
        logits = self.layer(batch_ctx)
        return logits
    
    def train_epoch(self, dl, optimiser, loss_fn):
        
        for batch in tqdm(dl):
            optimiser.zero_grad()
            contexts, words = batch
            logits = self.forward(contexts)
            
            loss = loss_fn(logits, words)
            loss.backward()

            optimiser.step()

    def train(self, num_epochs, lr = 0.1):
        optimiser = torch.optim.SGD(self.parameters(), lr = lr)
        loss_fn = nn.CrossEntropyLoss()
        train_dl = DataLoader(train_ds, batch_size = 128)
        dev_dl = DataLoader(dev_ds, batch_size = 128)

        for _ in range(num_epochs):
            self.train_epoch(train_dl, optimiser, loss_fn)
            loss = self.get_loss(dev_dl, loss_fn)
            print("Loss on validation set:", loss)
            perp = self.get_perp(dev_dl)
            print("Perp on validation set:", perp)

    def get_loss(self, dl, loss_fn):
        total = 0
        for batch in tqdm(dl):
            contexts, words = batch
            pred = self.forward(contexts)
            loss = loss_fn(pred, words)
            total += loss.item()

        return total

    def get_perp(self, dl):
        loss = nn.CrossEntropyLoss()
        sum_perp = 0
        count = 0
        for batch in tqdm(dl):
            contexts, words = batch
            pred = self.forward(contexts)

            cel = loss(pred, words).item()
            perp = math.exp(cel)

            sum_perp += perp
            count += len(batch)

        return (sum_perp/count)


## %%
#lm = LanguageModel(len(train_ds.vocab))
#lm.train(15)
#
## %%
#torch.save(lm, '10epochs.pth')
#
## %%
##lm = torch.load('10epochs.pth')
#test_dl = DataLoader(test_ds, batch_size = 128)
#perp = lm.get_perp(test_dl)
#print(perp)
#
## After 10 epochs, perplexities
## 109.83 on train
## 146.88 on val
## 147.13 on test
#

#751134
def perp_file(lm, inp_file, out_file, ds):
    lf = nn.CrossEntropyLoss()
    with open(inp_file, 'r') as f:
        with open(out_file, 'w') as g:
            for line in tqdm(f):
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
        
                try:
                    contexts = torch.stack(contexts)
                    words = torch.tensor(words)

                    outputs = lm(contexts)
                    loss = lf(outputs, words).item()
                    perp = math.exp(loss)

                    g.write(line[:-1] + '\t' + str(perp) + '\n')
                except RuntimeError: g.write('\n')

def hyper():
    lrs = [0.1, 0.01, 0.001]
    hs = [100, 300, 500]

    test_dl = DataLoader(test_ds, batch_size = 128)
    with open('hyper_rnn.txt', 'w') as f:
        for lr in lrs:
            for h in hs:
                lm = LanguageModel(len(train_ds.vocab), context_size = 4,
                                   hidden_size_1 = h, hidden_size_2 = h)
                print(f"lr={lr}-h={h}")
                lm.train(15, lr)

                p = lm.get_perp(test_dl)

                f.write(f"lr={lr}-n={n}: {p}\n")
