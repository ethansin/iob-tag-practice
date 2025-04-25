import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

def file_reader(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

def content_parser(file_content: str) -> list[list[str]]:
    parsed_datapoints = file_content.split("\n\n")
    tokenized_datapoints = [datapoint.split("\n") for datapoint in parsed_datapoints]
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

def tensorizer():
    pass

class NERDataset(Dataset):
    def __init__(self, dataset_filepath):
        file_contents = file_reader(dataset_filepath)
        self.utterances = content_parser(file_contents)

    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        sequence_tokens = []
        sequence_labels = []
        for token in self.utterances[idx]:
            parsed_annotation = token.split(" ")
            sequence_tokens.append(parsed_annotation[0])
            sequence_labels.append(parsed_annotation[-1])
        return sequence_tokens, sequence_labels

def main():
    train_dataset = NERDataset("dataset/ner.train")
    dev_dataset = NERDataset("dataset/ner.dev")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
    return

if __name__ == "__main__":
    main()