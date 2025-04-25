from utils.dataset_utils import NERDataset, read_ner_files, vocab_loader

def create_dataset(dataset_filepath, project_name="test"):
    text_dataset = read_ner_files(dataset_filepath)
    vocab = vocab_loader(project_name)
    dataset = NERDataset(text_dataset, vocab)

    return

if __name__ == "__main__":
    create_dataset("dataset/ner.train", "test")