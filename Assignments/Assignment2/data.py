from torchtext import vocab
from torch import tensor
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from tqdm import tqdm

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
        return self.tokens[index], self.labels[index]
    
    def __len__(self):
        return self.tokens.shape[0]

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

            self.contexts += [indices[:i] for i in range(1, len(indices))]
            self.words += indices[1:]

            indices = indices.reverse()
            self.contexts += [indices[:i] for i in range(1, len(indices))]
            self.words += indices[1:]

            self.max_length = len(words)-1 if len(words)-1 > self.max_length \
                              else self.max_length
        
        PAD_TOKEN = self.vocab['<PAD>']
        self.contexts = [ctx + [PAD_TOKEN] * (self.max_length - len(ctx))
                        for ctx in tqdm(self.contexts, desc="Padding")]
        
        self.contexts = tensor(self.contexts)
        self.words = tensor(self.words)
    
    def __getitem__(self, index):
        return self.contexts[index], self.words[index]
    
    def __len__(self):
        return self.contexts.shape[0]

processed_data = ProcessYelp("data/yelp-subset.dev.csv", 5)
#train_dataset = YelpData(vocab = processed_data.vocab, split = "train")
label_dev_dataset = LabelData(vocab = processed_data.vocab, split = "dev")
#nextword_dev_dataset = NextWordData(vocab = processed_data.vocab, split = "dev")