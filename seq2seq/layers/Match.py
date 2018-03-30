from __future__ import absolute_import

import keras
from keras import backend as K
from keras.engine import Layer


class Match(Layer):
    """
    Layer that computes a matching matrix between samples in two tensors.
    """

    def __init__(self, normalize=False, match_type='dot', embedding_size=None,
                 r=None, **kwargs):
        """

        Args:
            normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
            match_type:
            embedding_size:
            r:
            **kwargs:
        """
        super(Match, self).__init__(**kwargs)
        self.normalize = normalize
        self.match_type = match_type
        self.embedding_size = embedding_size
        self.r = r
        self.supports_masking = True
        if match_type not in ['dot', 'mul', 'plus', 'minus', 'concat',
                              'tensor2']:
            raise ValueError('In `Match` layer, '
                             'param match_type=%s is unknown.' % match_type)

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
        if self.match_type in ['tensor1', 'tensor2']:
            self.M = self.add_weight(name='tensor',
                                     shape=(
                                         self.embedding_size,
                                         self.embedding_size,
                                         self.r
                                     ),
                                     initializer=keras.initializers.RandomUniform(
                                         minval=-0.1, maxval=0.1, seed=None),
                                     trainable=True)
            self.W = self.add_weight(name='weight',
                                     shape=(2 * self.embedding_size, self.r),
                                     initializer=keras.initializers.RandomUniform(
                                         minval=-0.1, maxval=0.1, seed=None),
                                     trainable=True)
            self.b = self.add_weight(name='bias', shape=(1, self.r),
                                     initializer=keras.initializers.RandomUniform(
                                         minval=-0.1, maxval=0.1, seed=None),
                                     trainable=True)

    def call(self, inputs, **kwargs):
        x1 = inputs[0]
        x2 = inputs[1]
        # input: sample, timestep, 2*hidden
        if self.match_type in ['dot']:
            if self.normalize:
                x1 = K.l2_normalize(x1, axis=2)
                x2 = K.l2_normalize(x2, axis=2)
            output = K.tf.einsum('abd,acd->abc', x1, x2)
            output = K.tf.expand_dims(output, 3)
            print("Shape: ", output.shape)
            # sample, timestep1, timestep2, 1
        elif self.match_type in ['mul', 'plus', 'minus']:
            # x1 dim: sample, timestep1, timestep2, 2*hidden
            # x2 dim: sample, timestep1,timestep2, 2*hidden
            x1_exp = K.tf.stack([x1] * self.shape2[1], 2)
            x2_exp = K.tf.stack([x2] * self.shape1[1], 1)
            if self.match_type == 'mul':
                output = x1_exp * x2_exp
            elif self.match_type == 'plus':
                output = x1_exp + x2_exp
            elif self.match_type == 'minus':
                output = x1_exp - x2_exp
        elif self.match_type in ['concat']:
            x1_exp = K.tf.stack([x1] * self.shape2[1], axis=2)
            x2_exp = K.tf.stack([x2] * self.shape1[1], axis=1)
            # res dim: sample, timestep1,timestep2, 4*hidden
            output = K.tf.concat([x1_exp, x2_exp], axis=3)
        elif self.match_type in ['tensor1', 'tensor2']:
            x1 = K.tf.stack([x1] * self.shape2[1], 2)
            x2 = K.tf.stack([x2] * self.shape1[1], 1)
            part1 = K.tf.concat(
                [K.tf.reduce_sum(K.dot(x1, self.M[:, :, i]) * x2, axis=-1,
                                 keep_dims=True)
                 for i in range(self.r)],
                axis=-1)

            # part1 = [K.dot(x1,self.M[:,:,i]) *x2 for i in range(self.r)]

            part2 = K.dot(K.concatenate([x1, x2], -1), self.W)
            output = part1 + part2 + self.b
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Match` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if len(shape1) != 3 or len(shape2) != 3:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with 3 dimensions.')
        if shape1[0] != shape2[0] or shape1[2] != shape2[2]:
            raise ValueError('A `Match` layer should be called '
                             'on 2 inputs with same 0,2 dimensions.')

        if self.match_type in ['dot']:
            output_shape = [shape1[0], shape1[1], shape2[1], 1]
        elif self.match_type in ['mul', 'plus', 'minus']:
            output_shape = [shape1[0], shape1[1], shape2[1], shape1[2]]
        elif self.match_type in ['concat']:
            output_shape = [shape1[0], shape1[1], shape2[1],
                            shape1[2] + shape2[2]]
        elif self.match_type in ['tensor1', 'tensor2']:
            output_shape = [shape1[0], shape1[1], shape2[1], self.r]

        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'normalize': self.normalize,
            'match_type': self.match_type,
        }
        base_config = super(Match, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def match(inputs, axes, normalize=False, match_type='dot', **kwargs):
    """Functional interface to the `Match` layer.
    # Arguments
        inputs: A list of input tensors (with exact 2 tensors).
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the dot product matching matrix of the samples 
        from the inputs.
    """
    return Match(normalize=normalize, match_type=match_type, **kwargs)(inputs)
