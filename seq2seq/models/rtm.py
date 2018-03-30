# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras.layers import *
from keras.layers import Embedding

from model import BasicModel
from utils.utility import *


class RTM(BasicModel):
    def __init__(self, config):
        super(RTM, self).__init__(config)
        self.__name = 'RTM'
        self.check_list = []
        # self.check_list = ['text1_maxlen', 'text2_maxlen',
        #                    'embed', 'embed_size', 'train_embed', 'vocab_size',
        #                    'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[{0}] parameter check wrong'.format(self.__name))
        print('[{0}] init done'.format(self.__name), end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('hidden_size', 32)
        self.set_default('topk', 100)
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        show_layer_info('Input', doc)

        embedding = Embedding(self.config['vocab_size'],
                              self.config['embed_size'],
                              weights=[self.config['embed']],
                              trainable=self.embed_trainable)

        q_embed = embedding(Masking(mask_value=self.config['vocab_size'] - 1)(
            query))
        show_layer_info('Embedding', q_embed)
        d_embed = embedding(Masking(mask_value=self.config['vocab_size'] - 1)(
            doc))
        show_layer_info('Embedding', d_embed)

        q_rep = Bidirectional(
            LSTM(self.config['hidden_size'], return_sequences=True,
                 dropout=self.config['dropout_rate']))(q_embed)
        show_layer_info('Bidirectional-LSTM', q_rep)
        d_rep = Bidirectional(
            LSTM(self.config['hidden_size'], return_sequences=True,
                 dropout=self.config['dropout_rate']))(d_embed)
        show_layer_info('Bidirectional-LSTM', d_rep)

        # Output size: batch,timestep,2*hidden_size
