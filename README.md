# Stylistic Poetry
Developed by 清华大学人工智能研究院与社会人文计算研究中心. The codes basically come from our paper "Stylistic Chinese Poetry Generation via Unsupervised Style Disentanglement" (EMNLP 2018).

## Input files

vocab.pkl: The dictionary of character -> character id.

ivocab.pkl: The dictionary of character id -> character.

text_train.pkl, text_dev.pkl and text_test.pkl: Corpus files for training, validation and testing. Please refer to data/make_corpus.py for details about how to transform the poems into the required format.

## Main scripts

model.py: The implementation of the SPG model. The basis of this model is an LSTM encoder-decoder framework with attention mechanism.

state.py: The hyper-parameter settings.

train.py: The interface for training. Just setup the hyperparameters and use the command "python train.py".

generate.py: The interface for testing. Given the first sentence as input, a whole poem with four lines will be generated.

## Requirements
tensorflow-gpu 1.12

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