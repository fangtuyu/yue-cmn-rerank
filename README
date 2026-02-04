## yue-cmn-rerank

Source code and data for Escaping the Probability Trap: Mitigating Semantic Drift in Cantonese-Mandarin Translation（EACL 2026 Workshop LoResLM）

### Dependencies

- Python 3.8
- PyTorch 2.4.1 (with CUDA 12.1) 
- transformers 4.46.3
- sacrebleu 2.5.1
- numpy 1.24.2
- sentencepiece 0.2.0
- datasets 3.1.0
- accelerate 1.0.1

### Data

The datasets used is provided in `data/`. 

- Initial data in `data/original`

- Run to split data  `split_data.py  `

- > **Note**: split_data.py does not set a random seed, so the data splits will be different each time you run it.

- The data after splitting by split_data.py in `data/after_split`

- During the fine-tuning process, the segmented training data is further divided into a training set and a validation set, with the complete training set, validation set, and test set in `data/set`

### Run

Fine-tuning the forward (Cantonese-Mandarin) model

```
python finetune-fw.py
```

Fine-tuning the reverse (Mandarin-Cantonese) model

```
python finetune-rev.py
```

Generate candidate sentences , optimize weights based on the validation set, reranking the test set

```
python paramtuning-reranking.py
```

### Contact

If you have any questions regarding the code, please create an issue or contact the owner of this repository.
