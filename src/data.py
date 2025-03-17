import re
from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize

import torch
from torch.utils.data import Dataset


class Preprocessor:
    def __init__(self, vocab_size=10000, max_len=500):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2ind = {}
        self.ind2word = {}
        self.vocab = set()

    # токенизация отзывов и очистка от пунктуации
    def _tokenize(self, text):
        
        tokens = [
            word for sent in sent_tokenize(text)
            for word in word_tokenize(re.sub(r'[^\w\s]', '', sent.lower()))
        ]
        return tokens[:self.max_len]  

    def build_vocab(self, texts):
        words = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            words.update(tokens)

        special_tokens = ['<unk>', '<bos>', '<eos>', '<pad>'] # спец токены
        self.vocab = special_tokens + [token for token, _ in words.most_common(self.vocab_size)]

        # словарь для конвертации токена в индекс
        self.word2ind = {word: idx for idx, word in enumerate(self.vocab)}

        print(f"Размер словаря: {len(self.vocab)}")



class IMDBDataset(Dataset):
    def __init__(self, df, preprocessor):
        self.data = df
        self.preprocessor = preprocessor
        self.pad_id = preprocessor.word2ind['<pad>']
        self.bos_id = preprocessor.word2ind['<bos>']
        self.eos_id = preprocessor.word2ind['<eos>']

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['review']  

        # препроцессор для токенизации и индексации
        tokenized_review = self.preprocessor._tokenize(text)
        tokenized_review = [self.bos_id] + [
            self.preprocessor.word2ind.get(token, self.preprocessor.word2ind['<unk>'])
            for token in tokenized_review
        ] + [self.eos_id]

        sentiment = row['label']  
        rating = row['rating']   

        return {
            'review': torch.tensor(tokenized_review, dtype=torch.long),  
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'rating': torch.tensor(rating, dtype=torch.float)  
        }

    def __len__(self):
        return len(self.data)

