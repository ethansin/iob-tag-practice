import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Vocab():
    def __init__(self, text_dataset):
        self.token2idx = {
            "<pad>": 0,
            "<unk>": 1,
        }
        for utterance in text_dataset:
            for token in utterance[0]:
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.token2idx)
        self.idx2token = {idx:token for idx, token in self.token2idx.items()}

        self.label2idx = {
            "<pad>": 0,
            "<start>": 1,
            "<stop>": 2,
        }
        for utterance in text_dataset:
            for label in utterance[-1]:
                if label not in self.label2idx:
                    self.label2idx[label] = len(self.label2idx)
        self.label2idx = {idx:label for idx, label in self.label2idx.items()}

    def add_vocab(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)
            self.idx2token = {idx:token for idx, token in self.token2idx.items()}
    
    def length(self):
        return len(self.token2idx)

    def get_token_id(self, token):
        if token in self.token2idx:
            return self.token2idx[token]
        else:
            return 1
    
    def get_label_id(self, label):
        return self.label2idx[label]
    
class NERDataset(Dataset):
    def __init__(self, text_dataset: list, vocab: Vocab):
        self.token_tensors = []
        self.label_tensors = []

        for token_sequence, label_sequence in tqdm(text_dataset):
            self.token_tensors.append(tensorizer(token_sequence, vocab))
            self.label_tensors.append(tensorizer(label_sequence, vocab, is_label=True))

        if len(self.token_tensors) != len(self.label_tensors):
            raise ValueError

    def __len__(self):
        return len(self.token_tensors)
    
    def __getitem__(self, idx):
        token_tensor = self.token_tensors[idx]
        label_tensor = self.label_tensors[idx]

        return token_tensor, label_tensor
    
def collate_fn(batch):
    token_batch, label_batch = zip(*batch)
    batch_lengths = [len(sequence) for sequence in token_batch]
    return pad_sequence(token_batch, batch_first=True, padding_value=0), pad_sequence(label_batch, batch_first=True, padding_value=0), batch_lengths
    
def vocab_loader(project_name: str) -> Vocab:
    with open(f"projects/{project_name}/vocab.pkl", "rb") as f:
        return pickle.load(f)

def file_reader(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

def content_parser(file_content: str) -> list[list[str]]:
    parsed_datapoints = file_content.split("\n\n")
    tokenized_datapoints = []
    for datapoint in parsed_datapoints:
        tokenized_datapoints.append([token for token in datapoint.split("\n") if token != ''])
    return tokenized_datapoints

def read_ner_files(file_path: str) -> list:
    file_contents = file_reader(file_path)
    utterances = content_parser(file_contents)
    text_dataset = []
    for utterance in utterances:
        token_sequence = [token.split()[0] for token in utterance]
        label_sequence = [token.split()[-1] for token in utterance]
        text_dataset.append((token_sequence, label_sequence))
    return text_dataset

def tensorizer(sequence: list, vocab: Vocab, is_label: bool = False):
    output_tensor = torch.zeros(len(sequence), dtype=torch.long).to(device)
    if is_label:
        for idx in range(len(sequence)):
            output_tensor[idx] = vocab.get_label_id(sequence[idx])
    else:
        for idx in range(len(sequence)):
            output_tensor[idx] = vocab.get_token_id(sequence[idx])
    return output_tensor