import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, bidirectional=True, dropout=0.5):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)

        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):

        embedded = self.dropout(self.embedding(x))

        pack_embedded = pack_padded_sequence(embedded, lengths=lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(pack_embedded)

        output, _ = pad_packed_sequence(lstm_out, batch_first=True)

        logits = self.fc(self.dropout(output))

        return logits