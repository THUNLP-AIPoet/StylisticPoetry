# StylisticPoetry
Codes for Stylistic Chinese Poetry Generation via Unsupervised Style Disentanglement (EMNLP 2018)

## Input files

vocab.pkl: The dictionary of character -> character id.

ivocab.pkl: The dictionary of character id -> character.

text_train.pkl, text_dev.pkl and text_test.pkl: Corpus files for training, validation and testing. Please refer to data/make_corpus.py for details about how to transform the poems into the required format.

## Main scripts

model.py: The implementation of the SPG model.

train.py: The interface for training.

generate.py: The interface for testing.

state.py: The hyper-parameter settings.

## Cite
If you find this code useful for your research, please kindly cite this paper:

```
@inproceedings{yang2018stylistic,
  title={Stylistic Chinese Poetry Generation via Unsupervised Style Disentanglement},
  author={Yang, Cheng and Sun, Maosong and Yi, Xiaoyuan and Li, Wenhao},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={3960--3969},
  year={2018}
}
```