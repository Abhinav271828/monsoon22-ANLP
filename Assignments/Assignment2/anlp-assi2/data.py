from torchtext import vocab
from torch import tensor
from torch.utils.data import Dataset
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm
import random

DEVICE = torch.device('cuda:0')
class ProcessYelp():
    def __init__(self, filepath, min_freq):
        self.filepath = filepath
        self.min_freq = min_freq

        df = pd.read_csv(filepath)
        total_words = []
        for i in tqdm(range(len(df)), desc="Vocabulary"):
            line = df['text'][i]
            total_words += [[word.lower()] for word in word_tokenize(line)]
        
        self.vocab = vocab.build_vocab_from_iterator(total_words,
                                                     min_freq = min_freq,
                                                     specials = ['<UNK>', '<PAD>'])
        self.vocab.set_default_index(self.vocab['<UNK>'])

class LabelData(Dataset):
    def __init__(self, vocab : vocab.Vocab,
                       base_file_path : str = "data/yelp-subset.",
                       split : str = "train"):
        self.base_file_path = base_file_path
        self.vocab = vocab

        self.tokens = []
        self.labels = []

        self.max_length = 0

        self.pass_data(base_file_path + split + ".csv")

    def pass_data(self, filepath):
        df = pd.read_csv(filepath)
        for i in tqdm(range(len(df)), desc = "Tokenising"):
            self.labels.append(df['label'][i])

            line = df['text'][i]
            words = [word.lower() for word in word_tokenize(line)]
            indices = [self.vocab[word] for word in words]
            self.tokens.append(indices)

            self.max_length = len(words) if len(words) > self.max_length \
                              else self.max_length
        
        PAD_TOKEN = self.vocab['<PAD>']
        self.tokens = [sent + [PAD_TOKEN] * (self.max_length - len(sent))
                        for sent in tqdm(self.tokens, desc="Padding")]
        
        self.tokens = tensor(self.tokens)
        self.labels = tensor(self.labels)
    
    def __getitem__(self, index):
        return self.tokens[index].to(DEVICE), self.labels[index].to(DEVICE)
    
    def __len__(self):
        return len(self.tokens)

class NextWordData(Dataset):
    def __init__(self, vocab : vocab.Vocab,
                       base_file_path : str = "data/yelp-subset.",
                       split : str = "train"):
        self.base_file_path = base_file_path
        self.vocab = vocab

        self.contexts = []
        self.words = []

        self.max_length = 0

        self.pass_data(base_file_path + split + ".csv")

    def pass_data(self, filepath):
        df = pd.read_csv(filepath)
        for i in tqdm(range(len(df)), desc = "Tokenising"):
            line = df['text'][i]
            words = [word.lower() for word in word_tokenize(line)]
            indices = [self.vocab[word] for word in words]
            l = len(indices)

            self.contexts += [indices[:i] for i in range(1, l)]
            self.words += indices[1:]

            indices.reverse()
            self.contexts += [indices[:i] for i in range(1, l)]
            self.words += indices[1:]

            self.max_length = l-1 if l-1 > self.max_length \
                              else self.max_length
        
        zipped = random.sample(list(zip(self.contexts, self.words)),
                               int(len(self.contexts)/10))
        self.contexts = [c for c, _ in zipped]
        self.words = [w for _, w in zipped]

        PAD_TOKEN = self.vocab['<PAD>']
        self.contexts = [ctx + [PAD_TOKEN] * (self.max_length - len(ctx))
                        for ctx in tqdm(self.contexts, desc="Padding")]
        
        self.contexts = tensor(self.contexts)
        self.words = tensor(self.words)
    
    def __getitem__(self, index):
        return self.contexts[index].to(DEVICE), self.words[index].to(DEVICE)
    
    def __len__(self):
        return len(self.contexts)

processed_data = ProcessYelp("data/yelp-subset.train.csv", 5)
#label_train_dataset = LabelData(vocab = processed_data.vocab, split = "dev")
nextword_train_dataset = NextWordData(vocab = processed_data.vocab, split = "train")
