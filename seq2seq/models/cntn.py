# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from keras.layers import *
from keras.layers import Reshape, Embedding
from keras.models import Model
from layers.Bias import *
from layers.DynamicMaxPooling1D import *
from layers.Match import *
from layers.MatchTensorScore import *
from model import BasicModel
from utils.utility import *


class CNTN(BasicModel):
    def __init__(self, config):
        super(CNTN, self).__init__(config)
        self.__name = 'CNTN'
        self.check_list = ['text1_maxlen', 'text2_maxlen',
                           'embed', 'embed_size', 'vocab_size',
                           '1d_kernel_size', '1d_kernel_count',
                           'num_conv2d_layers', '2d_kernel_sizes',
                           '2d_kernel_counts', '2d_mpool_sizes',
                           'dropout_rate']
        self.embed_trainable = config['train_embed']
        self.setup(config)
        if not self.check():
            raise TypeError('[CNTN] parameter check wrong')
        print('[CNTN] init done', end='\n')

    def setup(self, config):
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)

        self.set_default('1d_kernel_count', 32)
        self.set_default('1d_kernel_size', 3)
        self.set_default('num_conv2d_layers', 2)
        self.set_default('2d_kernel_counts', [32, 32])
        self.set_default('2d_kernel_sizes', [[3, 3], [3, 3]])
        self.set_default('2d_mpool_sizes', [[3, 3], [3, 3]])
        self.set_default('dropout_rate', 0)
        self.config.update(config)

    def build(self):
        top_k = 5

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

        kc1 = 50
        ks1 = 3
        q_conv1 = Conv1D(kc1,
                         ks1, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            q_embed)
        show_layer_info('Conv1D', q_conv1)
        d_conv1 = Conv1D(kc1,
                         ks1, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            d_embed)
        show_layer_info('Conv1D', d_conv1)

        # sample,10,50
        shape_length1 = max(top_k,
                            math.ceil(1 / 3 * self.config['text1_maxlen']))
        shape_length2 = max(top_k,
                            math.ceil(1 / 3 * self.config['text2_maxlen']))
        q_pool1 = DynamicMaxPooling1D(top_k, shape_length1)(q_conv1)
        d_pool1 = DynamicMaxPooling1D(top_k, shape_length2)(d_conv1)
        show_layer_info('Pool1D', q_pool1)
        show_layer_info('Pool1D', d_pool1)
        q_nonlinear1 = Activation(activation='tanh')(Bias()(q_pool1))
        d_nonlinear1 = Activation(activation='tanh')(Bias()(d_pool1))

        kc2 = 100
        ks2 = 3
        # here we need another conv layer
        q_conv2 = Conv1D(kc2,
                         ks2, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            q_nonlinear1)
        d_conv2 = Conv1D(kc2,
                         ks2, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            d_nonlinear1)

        show_layer_info('Conv1D', q_conv2)
        show_layer_info('Conv1D', d_conv2)

        shape_length1 = max(top_k, math.ceil(2 / 3 * shape_length1))
        shape_length2 = max(top_k, math.ceil(2 / 3 * shape_length2))
        # here we need a max pooling layer
        q_pool2 = DynamicMaxPooling1D(top_k, shape_length1)(q_conv2)
        d_pool2 = DynamicMaxPooling1D(top_k, shape_length2)(d_conv2)

        show_layer_info('Pool1D', q_pool2)
        show_layer_info('Pool1D', d_pool2)

        q_nonlinear2 = Activation(activation='tanh')(Bias()(q_pool2))
        d_nonlinear2 = Activation(activation='tanh')(Bias()(d_pool2))

        kc3 = 200
        ks3 = 3
        q_conv3 = Conv1D(kc3,
                         ks3, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            q_nonlinear2)
        d_conv3 = Conv1D(kc3,
                         ks3, padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(
            d_nonlinear2)

        show_layer_info('Conv1D', q_conv3)
        show_layer_info('Conv1D', d_conv3)
        KMaxPooling1D = Lambda(
            lambda x:
            K.tf.transpose(
                K.tf.nn.top_k(
                    K.tf.transpose(
                        x,
                        [0, 2, 1]
                    ),
                    k=5,
                    sorted=True
                )[0],
                [0, 2, 1]
            )
        )

        q_pool3 = KMaxPooling1D(q_conv3)
        d_pool3 = KMaxPooling1D(d_conv3)

        show_layer_info('Pool1D', q_pool3)
        show_layer_info('Pool1D', d_pool3)

        q_out = Activation(activation='tanh')(Bias()(q_pool3))
        d_out = Activation(activation='tanh')(Bias()(d_pool3))

        qs_embed = Dense(50, kernel_regularizer=regularizers.l2(1e-4))(
            Reshape((-1,))(q_out))
        ds_embed = Dense(50, kernel_regularizer=regularizers.l2(1e-4))(
            Reshape((-1,))(d_out))

        show_layer_info('Dense', qs_embed)
        show_layer_info('Dense', ds_embed)

        out = MatchTensorScore(50, 5, 'relu',
                               kernel_regularizer=regularizers.l2(1e-4))(
            [qs_embed, ds_embed])

        show_layer_info("Tensor", out)
        model = Model(inputs=[query, doc], outputs=out)
        return model
