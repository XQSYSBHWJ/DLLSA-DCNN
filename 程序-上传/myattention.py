import keras as k
from keras import backend as K
import tensorflow as tf
import numpy as np


class MyAttention(k.layers.Layer):
    """注意力机制
    """

    def __init__(self, out_dim, key_size=8,  **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.key_size = key_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "out_dim": self.out_dim,
            "key_size": self.key_size,
        })
        return config

    def build(self, input_shape):
        super(MyAttention, self).build(input_shape)
        input_shape = list(input_shape)
        if input_shape[1] == None:
            input_shape[1] = 1
        kernel_initializer = 'glorot_uniform'
        kernel_regularizer = None
        kernel_constraint = None
        self.qw = self.add_weight(name='qw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.kw = self.add_weight(name='kw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.vw = self.add_weight(name='vw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)

    def call(self, inputs):
        input_size = tf.shape(inputs)
        q = tf.multiply(inputs, self.qw)
        k = K.permute_dimensions(tf.multiply(inputs, self.kw), (0, 2, 1))
        v = tf.multiply(inputs, self.vw)
        v = tf.reshape(tf.tile(v, [1, input_size[1], 1]), (input_size[0], input_size[1], input_size[1], self.out_dim))
        p = tf.matmul(q, k)
        p = tf.reshape(K.softmax(p / np.sqrt(self.key_size)), (input_size[0], input_size[1], input_size[1], 1))
        v = tf.reduce_sum(tf.multiply(v, p), 2)
        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.out_dim)

# if __name__ == '__main__':
# pass
