import torch
import torch.nn as nn

class IMDB_Model(nn.Module):

    def __init__(self, vocab_size, hidden_dim, n_layers, pad_idx, dropout, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout)
        self.normalization_sentiment = nn.LayerNorm(self.hidden_dim)

        self.fc_sentiment = nn.Linear(self.hidden_dim, 2)   # для тональности - бинарной классификации


    def forward(self, input_batch):
        embedding = self.embedding(input_batch)
        output, _ = self.rnn(embedding)

        hidden_sentiment = torch.mean(output, dim=1)
        hidden_sentiment = self.normalization_sentiment(hidden_sentiment)
        hidden_sentiment = self.dropout(hidden_sentiment)

        sentiment_logits = self.fc_sentiment(hidden_sentiment)

        return sentiment_logits
