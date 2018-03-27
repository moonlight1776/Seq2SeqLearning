# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras.layers import *
from keras.layers import Embedding
from keras.models import Model
from layers.Bias import *
from layers.Correlation import Correlation
from layers.DynamicMaxPooling1D import *
from layers.Match import *
from layers.MatchTensorScore import *
from model import BasicModel
from utils.utility import *


class WEC(BasicModel):
    def __init__(self, config):
        super(WEC, self).__init__(config)
        self.__name = 'WEC'
        self.check_list = []
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[WEC] parameter check wrong')
        print('[WEC] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

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
        # sample,10,50
        q_embed = embedding(query)
        show_layer_info('Embedding', q_embed)
        # sample,40,50
        d_embed = embedding(doc)
        show_layer_info('Embedding', d_embed)
        # abd,acd->abc
        # sample, 10,40
        score = Correlation(embedding_size=self.config['embed_size'])(
            [q_embed, d_embed])
        Score = Lambda(
            lambda x:
            K.tf.reduce_mean(K.tf.reduce_max(score, axis=-1), axis=-1,
                             keepdims=True)
        )
        out = Score(score)
        model = Model(inputs=[query, doc], outputs=out)
        return model
