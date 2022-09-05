# %% [markdown]
# Imports

# %%
import torch
from torch import tensor
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

device = torch.device('cpu')

# %%
with open('../../data/brown.txt', 'r') as f:
    para = ""
    sentences = []
    i = 0
    for line in tqdm(f, desc="Splitting dataset"):
        if (len(line) > 1): para += line[:-1]
        else:
            sentences += sent_tokenize(para)
            para = ""

shuffle(sentences)

split = (36597, 41825) # 70-10-20 of 52,282

with open('../../data/train.txt', 'w') as f: f.writelines([s + '\n' for s in sentences[:split[0]]])
with open('../../data/dev.txt', 'w') as f: f.writelines([s + '\n' for s in sentences[split[0]:split[1]]])
with open('../../data/test.txt', 'w') as f: f.writelines([s + '\n' for s in sentences[split[1]:]])

# %% [markdown]
# Code to read the embeddings

# %%
def get_embeddings(emb_file : str =  '/Volumes/Untitled/glove/glove.6B.300d.txt'):
    unk_emb_string = '0.22418134 -0.28881392 0.13854356 0.00365387 -0.12870757 0.10243822 0.061626635 0.07318011 -0.061350107 -1.3477012 0.42037755 -0.063593924 -0.09683349 0.18086134 0.23704372 0.014126852 0.170096 -1.1491593 0.31497982 0.06622181 0.024687296 0.076693475 0.13851812 0.021302193 -0.06640582 -0.010336159 0.13523154 -0.042144544 -0.11938788 0.006948221 0.13333307 -0.18276379 0.052385733 0.008943111 -0.23957317 0.08500333 -0.006894406 0.0015864656 0.063391194 0.19177166 -0.13113557 -0.11295479 -0.14276934 0.03413971 -0.034278486 -0.051366422 0.18891625 -0.16673574 -0.057783455 0.036823478 0.08078679 0.022949161 0.033298038 0.011784158 0.05643189 -0.042776518 0.011959623 0.011552498 -0.0007971594 0.11300405 -0.031369694 -0.0061559738 -0.009043574 -0.415336 -0.18870236 0.13708843 0.005911723 -0.113035575 -0.030096142 -0.23908928 -0.05354085 -0.044904727 -0.20228513 0.0065645403 -0.09578946 -0.07391877 -0.06487607 0.111740574 -0.048649278 -0.16565254 -0.052037314 -0.078968436 0.13684988 0.0757494 -0.006275573 0.28693774 0.52017444 -0.0877165 -0.33010918 -0.1359622 0.114895485 -0.09744406 0.06269521 0.12118575 -0.08026362 0.35256687 -0.060017522 -0.04889904 -0.06828978 0.088740796 0.003964443 -0.0766291 0.1263925 0.07809314 -0.023164088 -0.5680669 -0.037892066 -0.1350967 -0.11351585 -0.111434504 -0.0905027 0.25174105 -0.14841858 0.034635577 -0.07334565 0.06320108 -0.038343467 -0.05413284 0.042197507 -0.090380974 -0.070528865 -0.009174437 0.009069661 0.1405178 0.02958134 -0.036431845 -0.08625681 0.042951006 0.08230793 0.0903314 -0.12279937 -0.013899368 0.048119213 0.08678239 -0.14450377 -0.04424887 0.018319942 0.015026873 -0.100526 0.06021201 0.74059093 -0.0016333034 -0.24960588 -0.023739101 0.016396184 0.11928964 0.13950661 -0.031624354 -0.01645025 0.14079992 -0.0002824564 -0.08052984 -0.0021310581 -0.025350995 0.086938225 0.14308536 0.17146006 -0.13943303 0.048792403 0.09274929 -0.053167373 0.031103406 0.012354865 0.21057427 0.32618305 0.18015954 -0.15881181 0.15322933 -0.22558987 -0.04200665 0.0084689725 0.038156632 0.15188617 0.13274793 0.113756925 -0.095273495 -0.049490947 -0.10265804 -0.27064866 -0.034567792 -0.018810693 -0.0010360252 0.10340131 0.13883452 0.21131058 -0.01981019 0.1833468 -0.10751636 -0.03128868 0.02518242 0.23232952 0.042052146 0.11731903 -0.15506615 0.0063580726 -0.15429358 0.1511722 0.12745973 0.2576985 -0.25486213 -0.0709463 0.17983761 0.054027 -0.09884228 -0.24595179 -0.093028545 -0.028203879 0.094398156 0.09233813 0.029291354 0.13110267 0.15682974 -0.016919162 0.23927948 -0.1343307 -0.22422817 0.14634751 -0.064993896 0.4703685 -0.027190214 0.06224946 -0.091360025 0.21490277 -0.19562101 -0.10032754 -0.09056772 -0.06203493 -0.18876675 -0.10963594 -0.27734384 0.12616494 -0.02217992 -0.16058226 -0.080475815 0.026953284 0.110732645 0.014894041 0.09416802 0.14299914 -0.1594008 -0.066080004 -0.007995227 -0.11668856 -0.13081996 -0.09237365 0.14741232 0.09180138 0.081735 0.3211204 -0.0036552632 -0.047030564 -0.02311798 0.048961394 0.08669574 -0.06766279 -0.50028914 -0.048515294 0.14144728 -0.032994404 -0.11954345 -0.14929578 -0.2388355 -0.019883996 -0.15917352 -0.052084364 0.2801028 -0.0029121689 -0.054581646 -0.47385484 0.17112483 -0.12066923 -0.042173345 0.1395337 0.26115036 0.012869649 0.009291686 -0.0026459037 -0.075331464 0.017840583 -0.26869613 -0.21820338 -0.17084768 -0.1022808 -0.055290595 0.13513643 0.12362477 -0.10980586 0.13980341 -0.20233242 0.08813751 0.3849736 -0.10653763 -0.06199595 0.028849555 0.03230154 0.023856193 0.069950655 0.19310954 -0.077677034 -0.144811'
    unk_emb = tensor([float(x) for x in unk_emb_string.split()])
    embeddings = defaultdict(lambda : unk_emb)

    with open(emb_file, 'r') as f:
        for line in tqdm(f, desc="Reading embeddings"):
            split = line.strip().split()
            embeddings[split[0]] = \
                tensor([float(x) for x in split[1:]])

    return embeddings

embeddings = get_embeddings()

# %% [markdown]
# Dataset Class

# %%
class TextData(Dataset):
    def __init__(self, file_path           : str               = '../../data/train.txt',
                       pretrained_emb_dict : dict              = embeddings,
                       frequency_cutoff    : int               = 1,
                       context_size        : int               = 4,
                       vocab               : list              = None):
        self.file_path = file_path
        self.frequency_cutoff = frequency_cutoff
        self.context_size = context_size

        self.contexts = []
        self.words = []

        self.frequency_dictionary = defaultdict(lambda : 0)
        self.vocab = vocab if vocab else []

        self.words2indices = defaultdict(lambda : 0)
        self.embeddings = pretrained_emb_dict

        self.max_length = 0

        with open(self.file_path, 'r') as f:
            for line in tqdm(f, desc="Obtaining vocabulary and freq counts"):
                words = [word.lower() for word in word_tokenize(line)]
                if (not vocab): self.vocab += words
                for word in words: self.frequency_dictionary[word] += 1
                self.max_length = max(len(words), self.max_length)

            self.max_length -= 1
            print("max length", self.max_length)

            if (not vocab):
                self.vocab = list(set(self.vocab))
                self.vocab = [word for word in self.vocab
                                if self.frequency_dictionary[word] > self.frequency_cutoff]
                self.vocab.append('<unk>')
                self.vocab.insert(0, '<pad>')
            for i, w in enumerate(self.vocab):
                self.words2indices[w] = i

        embeddings_list = [torch.zeros(300)]
        for word in self.vocab[1:]:
            embeddings_list.append(self.embeddings[word])
        embeddings_list.append(self.embeddings['<unk>'])
        self.embeddings = torch.stack(embeddings_list)

        with open(self.file_path, 'r') as f:
            for line in tqdm(f, desc="Creating dataset"):
                words = [word.lower() for word in word_tokenize(line)]
                indices = [self.words2indices[word]
                                for word in words]
                #embeddings = torch.stack([self.embeddings[i] for i in indices])

                for i in range(1, len(words) - 1):
                    #context = torch.concat([torch.zeros(self.max_length-i, 300),
                    #                        embeddings[:i,:]], dim=0)
                    context = torch.concat([torch.zeros(self.max_length-i),
			      	      	    torch.tensor(indices[:i])]).long()
                    self.contexts.append(context)
                    self.words.append(indices[i])
        
        self.contexts = torch.stack(self.contexts).to(device)
        self.words = torch.tensor(self.words).to(device)
    
    def __getitem__(self, idx):
        return (self.contexts[idx], self.words[idx])

    def __len__(self):
        return len(self.contexts)

# %%
train_ds = TextData()
with open('../../data/vocab.txt', 'w') as f:
    for word in train_ds.vocab:
        f.write(word + '\n')
#dev_ds = TextData('../../data/dev.txt', vocab = train_ds.vocab)
test_ds = TextData('../../data/test.txt', vocab = train_ds.vocab)
