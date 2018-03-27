
---
A fork from <a href="https://github
.com/faneshion/seq2seq">faneshion/Matchzoo</a>, add some new model in Question 
Answering Field. We rename the repository to show our focus. 


## Installation
For usage examples, you can run
```
python seq2seq/main.py --phase train --model_file examples/toy_example/config/arci_ranking.config
python seq2seq/main.py --phase predict --model_file examples/toy_example/config/arci_ranking.config
```

## Overview
### Data Preparation
The data preparation module aims to convert dataset of different text matching tasks into a unified format as the input of deep matching models. Users provide datasets which contains pairs of texts along with their labels, and the module produces the following files.

+	**Word Dictionary**: records the mapping from each word to a unique identifier called *wid*. Words that are too frequent (e.g. stopwords), too rare or noisy (e.g. fax numbers) can be  filtered out by predefined rules.
+	**Corpus File**: records the mapping from each text to a unique identifier called *tid*, along with a sequence of word identifiers contained in that text. Note here each text is truncated or padded to a fixed length customized by users.
+	**Relation File**: is used to store the relationship between two texts, each line containing a pair of *tids* and the corresponding label.
+   **Detailed Input Data Format**: a detailed explaination of input data format can be found in seq2seq/data/example/readme.md.

### Model Construction
In the model construction module, we employ Keras library to help users build the deep matching model layer by layer conveniently. The Keras libarary provides a set of common layers widely used in neural models, such as convolutional layer, pooling layer, dense layer and so on. To further facilitate the construction of deep text matching models, we extend the Keras library to provide some layer interfaces specifically designed for text matching.

Moreover, the toolkit has implemented two schools of representative deep text matching models, namely representation-focused models and interaction-focused models [[Guo et al.]](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf).

### Training and Evaluation
For learning the deep matching models, the toolkit provides a variety of objective functions for regression, classification and ranking. For example, the ranking-related objective functions include several well-known pointwise, pairwise and listwise losses. It is flexible for users to pick up different objective functions in the training phase for optimization. Once a model has been trained, the toolkit could be used to produce a matching score, predict a matching label, or rank target texts (e.g., a document) against an input text.

## Benchmark Results:
Here, we adopt <a href="https://www.microsoft.com/en-us/download/details.aspx?id=52419">WikiQA</a> dataset for an example to inllustrate the usage of seq2seq. WikiQA is a popular benchmark dataset for answer sentence selection in question answering. We have provided <a href="./data/WikiQA/run_data.sh">a script</a> to download the dataset, and prepared it into the seq2seq data format. In the <a href="">models directory</a>, there are a number of configurations about each model for WikiQA dataset. 

Take the DRMM as an example. In training phase, you can run
```
python seq2seq/main.py --phase train --model_file examples/wikiqa/config/drmm_wikiqa.config
```
In testing phase, you can run
```
python seq2seq/main.py --phase predict --model_file examples/wikiqa/config/drmm_wikiqa.config
```

For detail of model comparasion, please refer to `./doc/cntnEqn.pdf`.
Here, the DRMM_TKS is a variant of DRMM for short text matching. Specifically, the matching histogram is replaced by a top-k maxpooling layer and the remaining part are fixed. 

## Model Detail:

1. DRMM

this model is an implementation of <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

- model file: models/drmm.py
- model config: models/drmm_ranking.config

---
2. MatchPyramid

this model is an implementation of <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

- model file: models/matchpyramid.py
- model config: models/matchpyramid_ranking.config

---
3. ARC-I

this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: models/arci.py
- model config: models/arci_ranking.config

---
4. DSSM

this model is an implementation of <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

- model file: models/dssm.py
- model config: models/dssm_ranking.config

---
5. CDSSM

this model is an implementation of <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

- model file: models/cdssm.py
- model config: models/cdssm_ranking.config

---
6. ARC-II

this model is an implementation of <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: models/arcii.py
- model config: models/arcii_ranking.config

---
7. MV-LSTM

this model is an implementation of <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

- model file: models/mvlstm.py
- model config: models/mvlstm_ranking.config

-------
8. aNMM

this model is an implementation of <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
- model file: models/anmm.py
- model config: models/anmm_ranking.config

-------
9. DUET

this model is an implementation of <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

- model file: models/duet.py
- model config: models/duet_ranking.config

---
10. K-NRM

this model is an implementation of <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>

- model file: models/knrm.py
- model config: models/knrm_ranking.config

---
11. models under development:

<a href="https://arxiv.org/abs/1604.04378">Match-SRNN</a>, <a href="https://arxiv.org/abs/1710.05649">DeepRank</a> ....




## Environment
* python2.7+
* tensorflow 1.2+
* keras 2.06+
* nltk 3.2.2+
* tqdm 4.19.4+
* h5py 2.7.1+
