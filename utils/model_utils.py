import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.dataset_utils import Vocab

class LSTM(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=0.5):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab.length(), embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)

        self.crf_transitions = nn.Parameter(
            torch.randn(len(vocab.label2idx), len(vocab.label2idx))
        )

        self.crf_transitions.data[vocab.label2idx["<start>"], :] = -10000
        self.crf_transitions.data[:, vocab.label2idx["<stop>"]] = -10000

        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, len(vocab.label2idx))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):

        embedded = self.dropout(self.embedding(x))

        pack_embedded = pack_padded_sequence(embedded, lengths=lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(pack_embedded)

        output, _ = pad_packed_sequence(lstm_out, batch_first=True)

        logits = self.fc(self.dropout(output))

        return logits