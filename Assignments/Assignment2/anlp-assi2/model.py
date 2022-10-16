from torch import nn
from torch.utils.data import DataLoader
import torch
from data import *

class ELMoPrediction(nn.Module):
    def __init__(self, no_layers : int = 2,
                       vocab : vocab.Vocab = processed_data.vocab,
                       hidden_size : int = 300):
        super().__init__()
        self.no_layers = no_layers
        self.vocab = vocab
        self.hidden_size = hidden_size

        self.get_embedding_layer(vocab, hidden_size)

        self.lstm = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size,
                            num_layers = no_layers, bidirectional = True, batch_first = True)
        self.weights = torch.rand(no_layers)
        self.softmax = nn.Softmax(dim=0)

        self.lm_head = nn.Linear(in_features = hidden_size * 2, out_features = len(vocab))
        self.cls_head = nn.Linear(in_features = hidden_size * 4, out_features = 5)

    def get_embedding_layer(self, vocabulary, embedding_dim):
        glove_vectors = vocab.GloVe(name = '6B', dim = embedding_dim)
        UNK_EMB = torch.mean(glove_vectors.vectors, dim=0)
        embeddings = []
        for word in tqdm(vocabulary.get_itos(), desc="Rearranging"):
            if word in glove_vectors.itos: embeddings.append(glove_vectors[word])
            else: embeddings.append(UNK_EMB)
        embeddings = torch.stack(embeddings)

        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=vocabulary['<PAD>'])

    def forward(self, batch):
        tokens,    _    = batch
        # [bz, l], [bz]

        embeddings = self.embedding(tokens)
        # [bz, l, hs]

        _, 		    (contextual_reps, _               ) = self.lstm(embeddings)
        # ([bz, l, 2 * hs], ([2 * n, bz, hs], [2 * n, bz, hs]))

        _, bz, hs = contextual_reps.shape
        contextual_reps = (contextual_reps.transpose(0,1) # [bz, 2 * n, hs]
                                          .view(bz, 2, self.no_layers, hs))

        contextual_reps = \
        torch.mul(contextual_reps,      # [bz, 2, n, hs]
                  (self.softmax(self.weights)     # [n]
                        .unsqueeze(1).unsqueeze(0).unsqueeze(0)
                                                  # [1,  1, n, 1]
                        .repeat(bz, 2, 1, hs))    # [bz, 2, n, hs]
                        .to(DEVICE))

        final_rep = (torch.sum(contextual_reps, dim = 2) # [bz, 2, hs]
                          .flatten(1,2))                 # [bz, 2 * hs]

        return final_rep

    def nextword_head(self, batch):
        final_rep = self(batch)
        # [bz, 2 * hs]
        return self.lm_head(final_rep)

    def classification_head(self, batch):
        final_rep_f = self(batch)
        # [bz, 2 * hs]
        final_rep_b = self((torch.flip(batch[0], [1]), batch[1]))
        # [bz, 2 * hs]
        return self.cls_head(torch.concat([final_rep_f, final_rep_b], dim=1))

    def train_epoch(self, dl, task):
        head = {"nextword": self.nextword_head, "label": self.classification_head}[task]
        sum_loss = 0
        for b in tqdm(dl, desc="Training"):
            preds = head(b)
            loss = self.loss_fn(preds, b[1])
            sum_loss += loss.item()

        return (sum_loss/len(dl))

    def train(self, dl, num_epochs, task):
        self.optim = torch.optim.SGD(self.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = self.vocab['<PAD>'])

        for e in range(num_epochs):
            avg_loss = self.train_epoch(dl, task)
            print(f"Avg. loss = {avg_loss}")

            torch.save(self, f"n_e={e}-l={avg_loss:.3}.pth")

mdl = ELMoPrediction(hidden_size = 100).to(DEVICE)
nw_dl = DataLoader(nextword_train_dataset, batch_size = 64)
#l_dl = DataLoader(label_dev_dataset, batch_size = 8)
#mdl.train(nw_dl, num_epochs=10, task="nextword")
#mdl.train(l_dl, num_epochs=10, task="label")
