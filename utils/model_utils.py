import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.dataset_utils import Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim, hidden_dim, batch_size=64, num_layers=1, bidirectional=True, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        
        self.vocab = vocab

        self.embedding = nn.Embedding(vocab.length(), embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=bidirectional, dropout=dropout)

        self.h_0 = torch.randn(num_layers * 2 if bidirectional else num_layers, batch_size, hidden_dim).to(device)
        self.c_0 = torch.randn(num_layers * 2 if bidirectional else num_layers, batch_size, hidden_dim).to(device)

        self.crf_transitions = nn.Parameter(
            torch.randn(self.vocab.tagset_size, self.vocab.tagset_size).to(device)
        )

        self.crf_transitions.data[vocab.label2idx["<start>"], :] = -10000
        self.crf_transitions.data[:, vocab.label2idx["<stop>"]] = -10000

        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, self.vocab.tagset_size)

        self.dropout = nn.Dropout(dropout)

    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return idx.item()
    
    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def get_lstm_feats(self, x, lengths):

        embedded = self.dropout(self.embedding(x))

        pack_embedded = pack_padded_sequence(embedded, lengths=lengths, batch_first=True, enforce_sorted=False)

        lstm_out, _ = self.lstm(pack_embedded, (self.h_0, self.c_0))

        output, _ = pad_packed_sequence(lstm_out, batch_first=True)

        #torch.tensor() with the size (batch_size, max_sentence_length, tagset_size)
        emission_scores = self.fc(self.dropout(output))

        return emission_scores      

    def forward_alg(self, batch: torch.Tensor, lengths: list[int]):
        alphas = []

        for i, sentence in enumerate(batch):

            feats = sentence

            init_alphas = torch.full((1, self.vocab.tagset_size), -10000.).to(device)
            init_alphas[0][self.vocab.label2idx["<start>"]] = 0.

            forward_var = init_alphas

            for feat in feats[:lengths[i]]:
                curr_alphas = []

                for next_tag in range(self.vocab.tagset_size):

                    # lstm likelihood of next_tag being the tag for feat, is the same regardless of transition from previous tag so it is the same score projected as shape (1, tagset_size)
                    emission_scores = feat[next_tag].view(1, -1).expand(1, self.vocab.tagset_size)

                    # score from transition table to each possible 
                    transition_scores = self.crf_transitions[next_tag].view(1, -1)
                    next_tag_var = forward_var + emission_scores + transition_scores

                    curr_alphas.append(self.log_sum_exp(next_tag_var).view(1))
                
                forward_var = torch.cat(curr_alphas).view(1, -1)

            terminal_var = forward_var + self.crf_transitions[self.vocab.label2idx["<stop>"]]
            alphas.append(self.log_sum_exp(terminal_var))

        return alphas
    
    def score_sentence(self, sentence, tags):
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.vocab.label2idx["<start>"]], dtype=torch.long).to(device), tags])
        for i, token in enumerate(sentence):
            score = score + self.crf_transitions[tags[i+1], tags[i]] + token[tags[i+1]]
        score = score + self.crf_transitions[self.vocab.label2idx["<stop>"], tags[-1]]
        return score
    
    def neg_log_likelihood(self, batch_sentences, batch_tags, lengths):
        batch_feats = self.get_lstm_feats(batch_sentences, lengths)
        forward_scores = self.forward_alg(batch_feats, lengths)
        gold_scores = []

        for i, sentence in enumerate(batch_feats):
            sentence = sentence[:lengths[i]]
            gold_score = self.score_sentence(sentence, batch_tags[i])
            gold_scores.append(gold_score)

        return [forward - gold for forward, gold in zip(forward_scores, gold_scores)]
    
    def viterbi_decode(self, batch, lengths):
        best_paths = []
        path_scores = []

        for i, sentence in enumerate(batch):
            backpointers = []
            init_viterbi = torch.full((1, self.vocab.tagset_size), -10000.).to(device)
            init_viterbi[0][self.vocab.label2idx["<start>"]] = 0

            forward_var = init_viterbi
            for feat in sentence[:lengths[i]]:
                timestep_backpointers = []
                timestep_viterbi = []

                for next_tag in range(self.vocab.tagset_size):
                    next_tag_var = forward_var + self.crf_transitions[next_tag]
                    best_tag_id = self.argmax(next_tag_var)
                    timestep_backpointers.append(best_tag_id)
                    timestep_viterbi.append(next_tag_var[0][best_tag_id].view(1))

                forward_var = (torch.cat(timestep_viterbi) + feat).view(1, -1)
                backpointers.append(timestep_backpointers)

            terminal_var = forward_var + self.crf_transitions[self.vocab.label2idx["<stop>"]]
            best_tag_id = self.argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            best_path = [best_tag_id]
            for timestep_backpointers in reversed(backpointers):
                best_tag_id = timestep_backpointers[best_tag_id]
                best_path.append(best_tag_id)

            start_tag = best_path.pop()
            assert start_tag == self.vocab.label2idx["<start>"]
            best_path.reverse()
            
            best_paths.append(best_path)
            path_scores.append(path_score)

        return path_scores, best_paths

    def forward(self, x, lengths, is_crf=False):
        
        if is_crf:
            lstm_feats = self.get_lstm_feats(x, lengths)

            scores, tag_sequences = self.viterbi_decode(lstm_feats, lengths)

            return scores, tag_sequences

        else:
            embedded = self.dropout(self.embedding(x))

            pack_embedded = pack_padded_sequence(embedded, lengths=lengths, batch_first=True, enforce_sorted=False)

            lstm_out, _ = self.lstm(pack_embedded, (self.h_0, self.c_0))

            output, _ = pad_packed_sequence(lstm_out, batch_first=True)

            logits = self.fc(self.dropout(output))

            return logits