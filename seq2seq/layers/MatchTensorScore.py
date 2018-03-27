from __future__ import absolute_import

import keras
from keras import activations
from keras import backend as K
from keras.engine import Layer


class MatchTensorScore(Layer):
    def __init__(self, embedding_size, r, activation, kernel_regularizer,
                 **kwargs):
        super(MatchTensorScore, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.r = r
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.M = self.add_weight(name='tensor',
                                 shape=(
                                     self.embedding_size,
                                     self.embedding_size,
                                     self.r
                                 ),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)
        self.V = self.add_weight(name='weight',
                                 shape=(self.r, 2 * self.embedding_size),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)
        self.b = self.add_weight(name='bias', shape=(1, self.r),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)
        self.u = self.add_weight(name='kernel', shape=(1, self.r),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)

    def call(self, inputs, **kwargs):
        # (sample, embedding)
        x1 = inputs[0]
        # (sample, embedding)
        x2 = inputs[1]

        # 
        part1 = K.tf.concat(
            [K.tf.reduce_sum(K.dot(x1, self.M[:, :, i]) * x2, axis=-1,
                             keepdims=True)
             for i in range(self.r)],
            axis=-1)
        print(K.tf.shape(part1))
        # r,
        part2 = K.tf.transpose(
            K.dot(self.V, K.tf.transpose(K.concatenate([x1, x2], -1))))
        output = part1 + part2
        output = output + self.b
        return K.tf.transpose(K.dot(self.u, K.tf.transpose(
            activations.get(self.activation)(output))))

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
