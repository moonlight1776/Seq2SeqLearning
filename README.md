# Seq2SeqLearning
## Introduction
A fork from <a href="https://github.com/faneshion/Matchzoo">faneshion/Matchzoo</a>, add some new model in Question Answering Field. We rename the repository to show our focus. 

## Environment
* Python 2.7+
* Tensorflow 1.2+
* Keras 2.06+
* nltk 3.2.2+
* tqdm 4.19.4+
* h5py 2.7.1+


## Overview
### Data Preparation
First please configure `nltk` resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

The data preparation module aims to convert dataset of different text matching tasks into a unified format as the input of deep matching models. Users provide datasets which contains pairs of texts along with their labels, and the module produces the following files.

+	**Word Dictionary**: records the mapping from each word to a unique identifier called *wid*. Words that are too frequent (e.g. stopwords), too rare or noisy (e.g. fax numbers) can be  filtered out by predefined rules.
+	**Corpus File**: records the mapping from each text to a unique identifier called *tid*, along with a sequence of word identifiers contained in that text. Note here each text is truncated or padded to a fixed length customized by users.
+	**Relation File**: is used to store the relationship between two texts, each line containing a pair of *tids* and the corresponding label.
+   **Detailed Input Data Format**: a detailed explaination of input data format can be found in `seq2seq/data/example/readme.md`.

Please run `bash ./data/WikiQA/run_data.sh` to get and format dataset.

### (Alternative) Custom dataset
#### Data processing

You should treat your data format as `sample.txt`, formatted as `label \t query\t document_txt`.In detail, `label` donotes the relation between `query` and `document`, `1` means the `query` is related to the `document`, otherwise it does not matter.the words in `query` and `documents` are separated by white space.To understand that most models require `corpus_preprocessed`.txt, `relation_train.txt`, 'relation_valid.txt', 'relation_test.txt', `embed_glove_d50` files,do the following:

1.Generating the files named 'relation_train.txt', 'relation_valid.txt', 'relation_test.txt' and 'corpus.txt' by excuting the file named `preparation.py` which can be found under `./seq2seq/inputs`. You can follow the process of production data below, which is the function 'main' of the 'preparation.py' script.

## begining of this process
1)create Preparation object.
2)specify the base directory where you want to load the files.
3)call the function 'run_with_one_corpus' of the Preparation object and specify a parameter, which is denotes the raw data.The function of 'run_with_one_corpus' is transfering the format of 'sample.txt' to the format like this 'id words', then outputing the relation files 'rel' between queries and doecuments.
4)save 'corpus.txt' by calling the function 'save_corpus' of Preparation object.
5)shuffle the relationship in the file 'rel', and then cut it into a specified percentage of the file. Detailedly, If you want to adjust output data radio, specify the seond parameter of function 'split_train_valid_test', and which represent the percentage of the training data, valid data, test data, orderly.
6)save relationship file by calling the function 'save_relation' of Preparation object, with specify the path you want to save.
## ending of this process


2.Generating the files named 'corpus_preprocessed.txt' , 'wodr_dict.txt'. And if you need CF,CF,IDF of documents,you can save 'word_stats.txt' by excuting function 'preprocessor.save_words_stats'. Amply, the models in MatchZoo requires that all words must be id instead of string, so you should map the string to id by excuting the function named 'preprocessor.run', then specify it's output name and save it.Generate the files , referencing to the function 'main' of the file 'preprocess.py' which can be found under MatchZoo/matchzoo/inputs.

## begining of this process
1)create Preprocess object, and you can specify the parameter making some constriant like spcify the frequence of words, which is filtered if the words do not in thei frequency band etc.
2)excute the function 'run' of object Preprocess, then get the documents' id and words mapped as id, with your inititialization paramters.
3)save word dict file by excuting function 'save_word_dict' of Preprocess object, and save 'word_stats.txt' by excuting function  'save_words_stats' of Preprocess object too, which contains information like DF,CF,IDF sequentially.
4)then save corpus information whose words has been mapped as id, with specified path by yourself.
## ending of this process


3.Generating the file named 'embed_glove_d50' by excuting the file named 'gen_w2v.py' which at the patch of MatchZoo MatchZoo/data/WikiQA/, whith three parameters 'glove.840B.300d.txt', 'word_dict.txt', 'embed_glove_d300', And the first parameter denotes the embedding download from url'http://nlp.stanford.edu/data/glove.840B.300d.zip' or where you want, the second parameter denotes the file mapping string words to words id, the last paramter denotes the output embedding for your own dataset starting with word id and following with embeddings.

## begining of this process
1)load word dict file and word vectors sequentially.
2)get the dimension of embedding you downloaded.
3)get the third parameter where is the path you want to output.
4)write the word embedding in your corpus.
5)randomly generated it, if the words in your corpus not in the embedding file you downloaded.
## ending of this process


4.What's more, here is how to generate special files for other models:

	4.1 Generating the files that DRMM needs. Generating the IDF file by cutting off a part of 'word_stats.txt' whith this command 'cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf' in linux console. Then generating histograms of DRMM. You can run this script 'python gen_hist4drmm.py 60'  as  WiKiQA data which is under  MatchZoo/data/WikiQA, and the only parameter represents the size of histograms.following this step found in function 'main' at file 'gen_hist4drmm.py', you can generate this file.

	## begining of this process
	1) get the size of histograms fisrt.
	2) specify the source file, and then define the path of the embedding file, relation file(including train, valid, and test), output file for histograms.
	3) random generate embedding vectors, then cover download embedding into it.The purpose of its doing so is to make words that are not in the dict also own random embedding to be used.
	4) read the corpus.
	5) generage histograms by calling function 'cal_hist' in the file 'preprocess ', which can be found in MatchZoo/matchzoo/inputs, then write it.
	## ending of this process


	4.2 Generating the files that ANMM needs by excuting 'gen_binsum4anmm.py' with only parameter that represents the number of bin, And you can found it under   MatchZoo/data/WikiQA also.it is similar to generating histograms for DRMM, the only difference is that function 'cal_hist' above is replaced by function 'cal_binsum'.

	4.3 Generating the files that DSSM and CDSSM require.You can refer to the code after line 66 of the file called 'prepare_mz_data.py' which is under the folder 'MatchZoo/data/WikiQA/'.the following is generating triletter  of words in corpus.

	## begining of this process
	1)define the input file 'word_dict.txt', output file 'triletter_dict.txt' and 'word_triletter_map.txt'. In detail, file 'triletter_dict.txt' contains information representing trilletter of words and its id, orderly. what's more, file 'word_triletter_map.txt' denotes words id and the id of the triletter it decomposed.
	2)read word dict.
	3)specify the begining and ending smybol of current word.
	4)split it to trilletter. call the function 'NgramUtil.ngrams' whose first parameter denotes the word will be split, second is how big granularity the word will be split, and last is the smybol of the connection string.
	5)fillter the too low and too high frequency words in the dict calling the function 'filter_triletter' by setting the minimum and maximum values.
	## ending of this process

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

    <a href="http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf">A Deep Relevance Matching Model for Ad-hoc Retrieval</a>.

- model file: `models/drmm.py`
- model config: `models/drmm_wikiqa.config`

---
2. MatchPyramid

    <a href="https://arxiv.org/abs/1602.06359"> Text Matching as Image Recognition</a>

- model file: `models/matchpyramid.py`
- model config: `models/matchpyramid_wikiqa.config`

---
3. ARC-I

    <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: `models/arci.py`
- model config: `models/arci_wikiqa.config`

---
4. DSSM

    <a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf">Learning Deep Structured Semantic Models for Web Search using Clickthrough Data</a>

- model file: `models/dssm.py`
- model config: `models/dssm_wikiqa.config`

---
5. CDSSM

    <a href="https://www.microsoft.com/en-us/research/publication/learning-semantic-representations-using-convolutional-neural-networks-for-web-search/">Learning Semantic Representations Using Convolutional Neural Networks for Web Search</a>

- model file: `models/cdssm.py`
- model config: `models/cdssm_wikiqa.config`

---
6. ARC-II

    <a href="https://arxiv.org/abs/1503.03244">Convolutional Neural Network Architectures for Matching Natural Language Sentences</a>

- model file: `models/arcii.py`
- model config: `models/arcii_wikiqa.config`

---
7. MV-LSTM

    <a href="https://arxiv.org/abs/1511.08277">A Deep Architecture for Semantic Matching with Multiple Positional Sentence Representations</a>

- model file: `models/mvlstm.py`
- model config: `models/mvlstm_wikiqa.config`

-------
8. aNMM

    <a href="http://maroo.cs.umass.edu/pub/web/getpdf.php?id=1240">aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model</a>
- model file: `models/anmm.py`
- model config: `models/anmm_wikiqa.config`

-------
9. DUET

    <a href="https://dl.acm.org/citation.cfm?id=3052579">Learning to Match Using Local and Distributed Representations of Text for Web Search</a>

- model file: `models/duet.py`
- model config: `models/duet_wikiqa.config`

---
10. K-NRM

    <a href="https://arxiv.org/abs/1706.06613">End-to-End Neural Ad-hoc Ranking with Kernel Pooling</a>

- model file: `models/knrm.py`
- model config: `models/knrm_wikiqa.config`

---
11. CNTN:

    <a href="https://www.ijcai.org/Proceedings/15/Papers/188.pdf">
Convolutional Neural Tensor Network Architecture for Community-Based Question Answering
</a>

- model file: `models/cntn.py`
- model config: `models/cntn_wikiqa.config`

---
12. (Experiment) GatedARC-II:
Modification on ARC-II with Gated Linear Unit

- model file: `models/gated_arcii.py`
- model config: `models/gated_arcii_wikiqa.config`

---





