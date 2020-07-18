import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2 # 采用双向的gru
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        # TODO:
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix], # embedding_matrix.shape = (30000, 256) - 3万个词，每个词256维
            trainable=False
        ) # TODO: 输入token找到token对应得词向量
        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        # TODO:
        
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True, # whether to return the last output in the output sequence
            return_state=True, # whether to return the last state in addition to the output
            recurrent_initializer='glorot_uniform'
        )
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1) # along dimension 1 将 hidden 分成两个tensor
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
