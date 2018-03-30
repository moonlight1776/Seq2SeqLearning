# -*- coding: utf8 -*-
from __future__ import print_function

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time

random.seed(49999)
import numpy as np

np.random.seed(49999)
import tensorflow

tensorflow.set_random_seed(49999)

from collections import OrderedDict

import keras
from keras.models import Model

from utils import *
import inputs
import metrics
from losses import *
from optimizers import *

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config=config)


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


def generate_data(dataset, input_conf):
    gen = OrderedDict()
    for tag, conf in input_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        gen[tag] = generator(config=conf)
    return gen


def collect_embedding(share_input_conf):
    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'],),
                                     dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2,
                                             [share_input_conf['vocab_size'],
                                              share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict,
                                                          embed=embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2,
                                             [share_input_conf['vocab_size'],
                                              share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')


def load_eval_metrics(config):
    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    return eval_metrics


def load_predict_dataset(input_conf):
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')
    return dataset


def process_predict_input_tags(input_conf, share_input_conf):
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (
        input_predict_conf.keys()), end='\n')
    return input_predict_conf


def train(config, pretrained_weights_filename=None):
    # add filemode="w" to overwrite

    train_time = time.time()
    print(json.dumps(config, indent=2), end='\n')
    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    optimizer = optimizers.get(optimizer)
    K.set_value(optimizer.lr, global_conf['learning_rate'])

    # If model comes from scratch, the default config will be used
    # If model has pretrained, we should load the weights by name
    path_list = str(global_conf['weights_file']).split('/')
    path_list.insert(-1, str(train_time))
    base_dir_list = path_list[:-1]
    base_dir = os.path.join(*base_dir_list)
    os.mkdir(base_dir)

    weight_dir = os.path.join(*path_list)
    log_path = os.path.join(base_dir, 'logs')
    weights_file = weight_dir + '.%d'
    config['restore_from'] = str(pretrained_weights_filename)
    with open(os.path.join(base_dir, '.config'), 'w') as fout:
        fout.write(json.dumps(config, indent=2))

    logging.basicConfig(filename=os.path.join(base_dir, '.log'),
                        level=logging.INFO)

    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    tb_call_back = keras.callbacks.TensorBoard(log_dir=log_path,
                                               histogram_freq=0,
                                               batch_size=32, write_graph=True,
                                               write_grads=False,
                                               write_images=False,
                                               embeddings_freq=0,
                                               embeddings_layer_names=None,
                                               embeddings_metadata=None)
    collect_embedding(share_input_conf)

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (
        input_train_conf.keys(), input_eval_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator

    train_gen = generate_data(dataset, input_train_conf)

    eval_gen = generate_data(dataset, input_eval_conf)

    ######### Load Model #########
    model = load_model_and_weight(config, pretrained_weights_filename)

    loss = []
    for lobj in config['losses']:
        if lobj['object_name'] in mz_specialized_losses:
            loss.append(
                rank_losses.get(lobj['object_name'])(lobj['object_params']))
        else:
            loss.append(rank_losses.get(lobj['object_name']))
    eval_metrics = load_eval_metrics(config)
    model.compile(optimizer=optimizer, loss=loss)
    print('[Model] Model Compile Done.', end='\n')

    init_iter = -1
    if pretrained_weights_filename is not None:
        init_iter = int(pretrained_weights_filename.split('.')[-1])
    for i_e in range(init_iter + 1, init_iter + 1 + num_iters):
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()

            debug_message1 = '[%s]\t[Train:%s] ' % (
                time.strftime('%m-%d-%Y %H:%M:%S',
                              time.localtime(time.time())),
                tag)

            history = model.fit_generator(
                genfun,
                steps_per_epoch=display_interval,
                epochs=1,
                shuffle=False,
                verbose=0,
                callbacks=[tb_call_back]
            )  # callbacks=[eval_map])
            debug_message2 = 'Iter:%d\tloss=%.6f' % (i_e, history.history[
                'loss'][0])
            debug_message = "{0}\t{1}".format(debug_message1, debug_message2)
            print(debug_message)
            logging.info(debug_message)

        for tag, generator in eval_gen.items():
            genfun = generator.get_batch_generator()
            debug_message1 = '[%s]\t[Eval:%s] ' % (
                time.strftime('%m-%d-%Y %H:%M:%S',
                              time.localtime(time.time())),
                tag)
            # print(debug_message)
            # logging.info(debug_message)
            res = dict([[k, 0.] for k in eval_metrics.keys()])
            num_valid = 0
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                if issubclass(type(generator),
                              inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts']
                    for k, eval_func in eval_metrics.items():
                        for lc_idx in range(len(list_counts) - 1):
                            pre = list_counts[lc_idx]
                            suf = list_counts[lc_idx + 1]
                            res[k] += eval_func(y_true=y_true[pre:suf],
                                                y_pred=y_pred[pre:suf])
                    num_valid += len(list_counts) - 1
                else:
                    for k, eval_func in eval_metrics.items():
                        res[k] += eval_func(y_true=y_true, y_pred=y_pred)
                    num_valid += 1
            generator.reset()
            debug_message2 = 'Iter:%d\t%s' % (
                i_e,
                '\t'.join(
                    ['%s=%f' % (k, v / num_valid) for k, v in res.items()]
                )
            )
            debug_message = "{0}\t{1}".format(debug_message1, debug_message2)
            print(debug_message)
            logging.info(debug_message)
            sys.stdout.flush()
        if (i_e + 1) % save_weights_iters == 0:
            model.save_weights(weights_file % (i_e + 1))


def predict(config, weights_file=None):
    ######## Read input config ########

    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    collect_embedding(share_input_conf)

    # list all input tags and construct tags config
    input_predict_conf = process_predict_input_tags(input_conf,
                                                    share_input_conf)

    # collect dataset identification
    dataset = load_predict_dataset(input_conf)

    # initial data generator
    predict_gen = generate_data(dataset, input_predict_conf)

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]

    model = load_model_and_weight(config, weights_file)

    eval_metrics = load_eval_metrics(config)
    res = dict([[k, 0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Predict] @ %s ' % (
            time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())),
            tag),
              end='')
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            y_pred = model.predict(input_data, batch_size=len(y_true))

            if issubclass(type(generator),
                          inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts) - 1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx + 1]
                        res[k] += eval_func(y_true=y_true[pre:suf],
                                            y_pred=y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts) - 1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx + 1]
                    for p, y, t in zip(input_data['ID'][pre:suf],
                                       y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_func(y_true=y_true, y_pred=y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d: d[1][0],
                                       reverse=True)
                        for inum, (did, (score, gt)) in enumerate(dinfo):
                            f.write('%s\tQ0\t%s\t%d\t%f\t%s\t%s\n' % (
                                qid, did, inum, score, config['net_name'], gt))
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d: d[1][0],
                                       reverse=True)
                        for inum, (did, (score, gt)) in enumerate(dinfo):
                            f.write('%s %s %s %s\n' % (gt, qid, did, score))

        print('[Predict] results: ', '\t'.join(
            ['%s=%f' % (k, v / num_valid) for k, v in res.items()]), end='\n')
        sys.stdout.flush()


# model_and_weight are tuple of tuple
# ((config1,weight1),(config2,weight2),...)
def ensemble_predict(mw_list):
    for item in mw_list:
        global_config = item[0]
        print(json.dumps(global_config, indent=2), end='\n')
        input_conf = global_config['inputs']
        share_input_conf = input_conf['share']

        # format data
        collect_embedding(share_input_conf)
        input_predict_conf = process_predict_input_tags(input_conf,
                                                        share_input_conf)

        dataset = load_predict_dataset(input_conf)
        predict_gen = generate_data(dataset, input_predict_conf)
        eval_metrics = load_eval_metrics(global_config)
        res = dict([[k, 0.] for k in eval_metrics.keys()])
        model = load_model_and_weight(item[0], item[1])
        for tag, generator in predict_gen.items():
            genfun = generator.get_batch_generator()
            print('[%s]\t[Predict] @ %s ' % (
                time.strftime('%m-%d-%Y %H:%M:%S',
                              time.localtime(time.time())),
                tag),
                  end='')
            num_valid = 0
            res_scores = {}
            total_length = 0
            for input_data, y_true in genfun:
                total_length += len(y_true)
                y_pred = model.predict(input_data, batch_size=len(y_true))

                if issubclass(type(generator),
                              inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts']
                    for k, eval_func in eval_metrics.items():
                        for lc_idx in range(len(list_counts) - 1):
                            pre = list_counts[lc_idx]
                            suf = list_counts[lc_idx + 1]
                            res[k] += eval_func(y_true=y_true[pre:suf],
                                                y_pred=y_pred[pre:suf])

                    y_pred = np.squeeze(y_pred)
                    for lc_idx in range(len(list_counts) - 1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx + 1]
                        for p, y, t in zip(input_data['ID'][pre:suf],
                                           y_pred[pre:suf], y_true[pre:suf]):
                            if p[0] not in res_scores:
                                res_scores[p[0]] = {}
                            res_scores[p[0]][p[1]] = (y, t)

                    num_valid += len(list_counts) - 1
                else:
                    for k, eval_func in eval_metrics.items():
                        res[k] += eval_func(y_true=y_true, y_pred=y_pred)
                    for p, y, t in zip(input_data['ID'], y_pred, y_true):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y[1], t[1])
                    num_valid += 1

            generator.reset()

            with open(os.path.join('.', 'best_model', str.lower(
                    global_config['net_name']) + '_score.pkl'),
                      'wb') as \
                    fout:
                pickle.dump(res_scores, fout)

            print('[Predict] results: ', '\t'.join(
                ['%s=%f' % (k, v / num_valid) for k, v in res.items()]),
                  end='\n')
            sys.stdout.flush()


def load_model_and_weight(config, weights_file):
    model = load_model(config)
    if weights_file is not None:
        model.load_weights(weights_file)
    return model


def load_config(config_path):
    with open(config_path, 'r') as fin:
        return json.load(fin)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train',
                        help='Phase: Can be train or predict, '
                             'the default value is train.')
    parser.add_argument('--model_file', default='./models/arci.config',
                        help='Model_file: Model file for the chosen model.')
    parser.add_argument('--weights_file', default=None,
                        help='Weight file: Weight file for the '
                             'chosen model.')

    args = parser.parse_args()
    model_config_file = args.model_file
    weights_file = args.weights_file

    if args.phase == 'train':
        model_config = load_config(model_config_file)
        train(model_config, weights_file)
    elif args.phase == 'predict':
        model_config = load_config(model_config_file)
        predict(model_config, weights_file)
    elif args.phase == 'ensemble_predict':
        model_name_list = ['arcii', 'cntn', 'mvlstm', 'wec']

        if not os.path.exists('best_model'):
            print('Cannot find trained model!')
            exit(-1)

        config_path = os.path.join('.', 'best_model', '{0}.config')
        weight_path = os.path.join('.', 'best_model', '{0}.weights')
        model_list = [(load_config(config_path.format(i)),
                       weight_path.format(i)) for
                      i in model_name_list]
        ensemble_predict(model_list)

    else:
        print('Phase Error.', end='\n')
    return


if __name__ == '__main__':
    main(sys.argv)
