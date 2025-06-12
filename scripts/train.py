import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from utils.dataset_utils import NERDataset, read_ner_files, vocab_loader, collate_fn
from utils.model_utils import BiLSTM_CRF
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

def create_dataset(dataset_filepath, project_name="test"):
    text_dataset = read_ner_files(dataset_filepath)
    vocab = vocab_loader(project_name)
    dataset = NERDataset(text_dataset, vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    
    return dataloader

def lstm_training_loop(train_dataloader: DataLoader, val_dataloader: DataLoader, project_name="test"):

    vocab = vocab_loader(project_name)
    embedding_dim = 50
    hidden_dim = 128

    model = BiLSTM_CRF(vocab, embedding_dim, hidden_dim).to(device)

    num_epochs = 25
    lr = 5e-4

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=lr)

    prev_accuracy = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_seqs, batch_labels, batch_lengths in tqdm(train_dataloader):
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_seqs, batch_lengths)

            outputs = outputs.view(-1, len(vocab.label2idx))
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

        if accuracy < prev_accuracy:
            break
        prev_accuracy = accuracy

def crf_lstm_training_loop(train_dataloader: DataLoader, val_dataloader: DataLoader, project_name="test"):
    
    vocab = vocab_loader(project_name)
    embedding_dim = 100
    hidden_dim = 128

    model = BiLSTM_CRF(vocab, embedding_dim, hidden_dim, num_layers=2).to(device)

    num_epochs = 15
    lr = 5e-4

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_running_loss = 0.0
        running_loss = []

        for batch_seqs, batch_labels, batch_lengths in tqdm(train_dataloader):
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)            
            model.zero_grad()
            losses = model.neg_log_likelihood(batch_seqs, batch_labels, batch_lengths)
            loss = sum(losses) / len(losses)

            loss.backward()
            optimizer.step()

            running_loss.append(loss)

        epoch_running_loss = (sum(running_loss) / len(running_loss)).item()
    
        print(f"\n--------------------------\nEpoch {epoch+1} Train Loss: {epoch_running_loss}\n--------------------------\n")

        with torch.no_grad():

            all_predictions = []
            all_gold_labels = []

            is_crf = True

            for batch_seqs, batch_labels, batch_lengths in tqdm(val_dataloader):
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels
                _, prediction_tagseqs = model(batch_seqs, batch_lengths, is_crf)

                for i, prediction in enumerate(prediction_tagseqs):
                    all_predictions.append(prediction)
                    all_gold_labels.append(batch_labels[i][:batch_lengths[i]])

            flattened_predictions = [tag for tags in all_predictions for tag in tags]
            flattened_gold_labels = [tag for tags in all_gold_labels for tag in tags]

            f1 = f1_score(flattened_gold_labels, flattened_predictions, average='weighted', zero_division=np.nan)
            precision = precision_score(flattened_gold_labels, flattened_predictions, average='weighted', zero_division=np.nan)
            recall = recall_score(flattened_gold_labels, flattened_predictions, average='weighted', zero_division=np.nan)

        print(f"\n-------------------\nEpoch {epoch+1} Val Metrics\n-------------------\nF1: {f1}\nPRECISION: {precision}\nRECALL: {recall}\n-------------------\n")

    torch.save(model.state_dict(), "projects/test/model_weights.pth")

if __name__ == "__main__":
    train_dataloader = create_dataset("dataset/ner.train", "test")
    val_dataloader = create_dataset("dataset/ner.dev", "test")

    crf_lstm_training_loop(
        train_dataloader,
        val_dataloader,
        "test"
    )