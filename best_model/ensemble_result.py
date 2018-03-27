import os
import pickle
import sys

sys.path.append('..')
from seq2seq.utils import *

word_dict, _ = read_word_dict(
    os.path.join('.', 'data', 'WikiQA', 'word_dict.txt'))
sentence_list, _ = read_data(
    os.path.join('.', 'data', 'WikiQA', 'corpus_preprocessed.txt'))


def id2sentence(id):
    if word_dict is not None and sentence_list is not None:
        return ' '.join([word_dict[word_id] for word_id in sentence_list[id]])


def load_score(name):
    with open(name, 'rb') as fin:
        return pickle.load(fin)


def find_suspect_pair(score):
    sus_pair = dict()
    for k, v in score.items():
        max_value = -100
        correct_key = None
        max_key = None
        for k1 in v:
            if v[k1][1] == 1:
                correct_key = k1
            if v[k1][0] > max_value:
                max_value = v[k1][0]
                max_key = k1
        if correct_key is not None and correct_key != max_key:
            sus_pair[k] = (
                (correct_key, v[correct_key]), (max_key, v[max_key]))
    return sus_pair


def build_id_db(key_db):
    # Every misplaced question id with four candidate answers
    db = dict()
    for question_id in key_db:
        correct_key = None
        candidate_answers = []
        candidate_answers.append(correct_key)
        for i in pair_list:
            if question_id not in i:
                candidate_answers.append(correct_key)
            else:
                correct_key = i[question_id][0][0]
                candidate_answers.append(i[question_id][1][0])
        db[question_id] = tuple(
            [x if x is not None else correct_key for x in candidate_answers])
    return db


def build_key_set(p_list):
    key_set = set()
    for i in p_list:
        for question_id in i:
            key_set.add(question_id)
    return key_set


name_list = ['WEC', 'CNTN', 'MVLSTM', 'ARCII']

score_filename_list = [os.path.join('best_model', i.lower() + '_score.pkl') for
                       i in name_list]
score_list = [load_score(name) for name in score_filename_list]
pair_list = [find_suspect_pair(score) for score in score_list]

k_set = build_key_set(pair_list)
id_db = build_id_db(k_set)
with open(os.path.join('best_model', 'result.txt'), 'wb') as fout:
    for question_id in id_db:
        fout.write(id2sentence(question_id).encode('utf-8'))
        for answer_id in id_db[question_id]:
            fout.write(('\n' + id2sentence(answer_id)).encode('utf-8'))
        fout.write('\n\n'.encode('utf-8'))
