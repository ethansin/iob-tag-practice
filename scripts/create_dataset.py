import torch
import torch.nn as nn
from torch.optim import Adam
from utils.dataset_utils import NERDataset, read_ner_files, vocab_loader, collate_fn
from utils.model_utils import LSTM
from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

def create_dataset(dataset_filepath, project_name="test"):
    text_dataset = read_ner_files(dataset_filepath)
    vocab = vocab_loader(project_name)
    dataset = NERDataset(text_dataset, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    return dataloader

def training_loop(train_dataloader: DataLoader, val_dataloader: DataLoader, project_name="test"):

    vocab = vocab_loader(project_name)
    vocab_size = vocab.length()
    embedding_dim = 50
    hidden_dim = 128
    output_dim = len(vocab.label2idx)

    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)

    num_epochs = 1
    lr = 1e-5

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_seqs, batch_labels, batch_lengths in tqdm(train_dataloader):
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_seqs, batch_lengths)

            outputs = outputs.view(-1, output_dim)
            batch_labels = batch_labels.view(-1)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_seqs, batch_labels, batch_lengths in tqdm(val_dataloader):
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_seqs, batch_lengths)
                predictions = torch.argmax(outputs, dim=-1)

                mask = batch_labels != 0
                correct += (predictions == batch_labels).masked_select(mask).sum().item()
                total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_dataloader = create_dataset("dataset/ner.train", "test")
    val_dataloader = create_dataset("dataset/ner.dev", "test")

    training_loop(
        train_dataloader,
        val_dataloader,
        "test"
    )