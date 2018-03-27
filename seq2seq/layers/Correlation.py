from __future__ import absolute_import

import keras
from keras import backend as K
from keras.engine import Layer


class Correlation(Layer):
    """Layer that computes a matching matrix between samples in two tensors.
    # Arguments
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, normalize=False, match_type='dot', embedding_size=None,
                 r=None, **kwargs):
        super(Correlation, self).__init__(**kwargs)
        self.normalize = normalize
        self.match_type = match_type
        self.embedding_size = embedding_size
        self.r = r
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        self.shape1 = input_shape[0]
        self.shape2 = input_shape[1]
        if self.shape1[0] != self.shape2[0]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (self.shape1[0], self.shape2[0]) +
                'Layer shapes: %s, %s' % (self.shape1, self.shape2))
        if self.shape1[2] != self.shape2[2]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (self.shape1[2], self.shape2[2]) +
                'Layer shapes: %s, %s' % (self.shape1, self.shape2))
        # We fisrt use idea in second paper
        self.M = self.add_weight(name='weight',
                                 shape=(
                                     self.embedding_size, self.embedding_size),
                                 initializer=keras.initializers.RandomUniform(
                                     minval=-0.1, maxval=0.1, seed=None),
                                 trainable=True,
                                 regularizer=keras.regularizers.l2(1e-4))

    def call(self, inputs, **kwargs):
        x1 = inputs[0]
        x2 = inputs[1]
        x1 = K.tf.tensordot(x1, self.M, axes=[[2], [0]])
        if self.normalize:
            x1 = K.l2_normalize(x1, axis=2)
            x2 = K.l2_normalize(x2, axis=2)
        output = K.tf.einsum('abd,acd->abc', x1, x2)
        print("Shape: ", output.shape)
        # sample, timestep1, timestep2, 1
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.embedding_size

    def compute_mask(self, inputs, mask=None):
        return None
