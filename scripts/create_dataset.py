import torch
from utils.dataset_utils import NERDataset, read_ner_files, vocab_loader, collate_fn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

def create_dataset(dataset_filepath, project_name="test"):
    text_dataset = read_ner_files(dataset_filepath)
    vocab = vocab_loader(project_name)
    dataset = NERDataset(text_dataset, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch)
        break

    return

if __name__ == "__main__":
    create_dataset("dataset/ner.train", "test")