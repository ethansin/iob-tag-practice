Practice building a BiLSTM CRF completely from scratch using only PyTorch.

The model architecture including the forward algorithm and viterbi decoding steps are in a module defined in `utils/model_utils`.

The training loop is found in `scripts/train.py`. 

If you want to test my code (on macOS): 

```
cd path/to/iob-tag-practice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="./"
python3 scripts/build_vocab.py --train_dataset_filepath dataset/ner.train --project_name test
python3 scripts/train.py
```

This only checks for assigning tensors to `mps` as I was developing this locally on my M1 device. If you want to use `cuda` then you will need to change the device variable to check for `cuda` in `dataset_utils`, `model_utils`, and `train.py`. 