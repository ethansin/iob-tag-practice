import pickle
import argparse
from pathlib import Path
from utils.dataset_utils import Vocab, read_ner_files


def main(train_dataset_filepath, project_name):

    folder = Path(f'projects/{project_name}')
    folder.mkdir(parents=True, exist_ok=True)

    train_text_dataset = read_ner_files(train_dataset_filepath)

    vocab = Vocab(train_text_dataset)

    with open(f"{folder}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_filepath', type=str, help='path to your NER train dataset filepath')
    parser.add_argument('--project_name', type=str, help='id for your current project')

    args = parser.parse_args()

    main(**vars(args))