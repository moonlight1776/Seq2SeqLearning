# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras.layers import *
from keras.layers import Reshape, Embedding
from keras.models import Model
from layers.Match import *
from model import BasicModel
from utils.utility import *


class MVLSTM(BasicModel):
    def __init__(self, config):
        super(MVLSTM, self).__init__(config)
        self.__name = 'MVLSTM'
        self.check_list = ['text1_maxlen', 'text2_maxlen',
                           'embed', 'embed_size', 'train_embed', 'vocab_size',
                           'hidden_size', 'topk', 'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[MVLSTM] parameter check wrong')
        print('[MVLSTM] init done', end='\n')

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
        # Output size: sample,timestep,2*hidden_num
        cross = Match(match_type=self.config['match_type'],
                      embedding_size=2 * self.config['hidden_size'], r=5)(
            [q_rep, d_rep])
        # cross = Dot(axes=[2, 2])([q_embed, d_embed])
        show_layer_info('Match', cross)

        if self.config['match_type'] != 'tensor2':
            cross_reshape = Reshape((-1,))(cross)
            show_layer_info('Reshape', cross_reshape)

            mm_k = Lambda(
                lambda x: K.tf.nn.top_k(x, k=self.config['topk'], sorted=True)[
                    0])(cross_reshape)
            show_layer_info('Lambda-topk', mm_k)

            pool1_flat_drop = Dropout(rate=self.config['dropout_rate'])(mm_k)
            show_layer_info('Dropout', pool1_flat_drop)
        else:
            act_cross = Activation('relu')(cross)
            pool1_flat_drop = Lambda(
                lambda x:
                K.tf.reshape(
                    K.tf.nn.top_k(
                        K.tf.transpose(
                            K.tf.reshape(
                                x,
                                (-1, x.shape[1] * x.shape[2], x.shape[3])
                            ),
                            [0, 2, 1]
                        ),
                        k=self.config['topk'],
                        sorted=True
                    )[0],
                    (-1,
                     K.tf.Dimension(self.config['topk'] * x.shape[3].value)
                     )
                )
            )(act_cross)

        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax')(pool1_flat_drop)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1)(pool1_flat_drop)
        show_layer_info('Dense', out_)

        # model = Model(inputs=[query, doc, dpool_index], outputs=out_)
        model = Model(inputs=[query, doc], outputs=out_)
        return model
