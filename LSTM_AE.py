import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell

import numpy as np

class LSTM_AE(object):

    def __init__(self, hidden_num, inputs, step_num):
        self.input_ = inputs
        self.prob = tf.placeholder_with_default(1.0, shape=()) #for dropout
        #inputs = tf.nn.dropout(inputs, self.prob) #turn on/off for dropout
        inputs = [tf.squeeze(t, [1]) for t in tf.split(inputs, step_num, 1)]
        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        
        cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
        self._enc_cell = cell
        self._dec_cell = cell

        with tf.variable_scope('encoder'):
            (self.z_codes, self.enc_state) = tf.contrib.rnn.static_rnn(self._enc_cell, inputs, dtype=tf.float32)

        with tf.variable_scope('decoder') as vs:
            dec_weight_ = tf.Variable(tf.truncated_normal([hidden_num, self.elem_num], dtype=tf.float32), name='dec_weight')
            dec_bias_ = tf.Variable(tf.constant(0.1, shape=[self.elem_num], dtype=tf.float32), name='dec_bias')

            dec_state = self.enc_state
            dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
            dec_outputs = []
            for step in range(len(inputs)):
                if step > 0:
                    vs.reuse_variables()
                (dec_input_, dec_state) = self._dec_cell(dec_input_, dec_state)
                dec_input_ = tf.matmul(dec_input_, dec_weight_) + dec_bias_
                dec_outputs.append(dec_input_)
            dec_outputs = dec_outputs[::-1]
            self.output_ = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))
        self.train = tf.train.AdamOptimizer().minimize(self.loss)
        
    def get_enc_cell(self):
        enc_weights = [v for v in tf.global_variables() if v.name == 'encoder/rnn/lstm_cell/kernel:0'][0]
        return enc_weights.eval()
