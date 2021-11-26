import tensorflow as tf
from tensorflow import keras
import numpy as np

class Encoder(keras.models.Model):
  def __init__(self, n_units, vocab_size, embedding_dim):
    super(Encoder, self).__init__()
    self.enc_units = n_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
  
  def call(self, x, hidden):
    x = self.embedding(x)
    x, hidden = self.gru(x, initial_state=hidden)
    return x, hidden

  def initialize_hidden_state(self, batch_size):
    return tf.zeros((batch_size, self.enc_units))
# -----------------------------------------------------------------------------------------------------------
class BahdanauAttention(keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(query) + self.W2(values))) # score.shape = (b, t, 1)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
# -------------------------------------------------------------------------------------------------------------
class Decoder(keras.models.Model):
  def __init__(self, n_units, vocab_size, embedding_dim):
    super(Decoder, self).__init__()
    self.dec_units = n_units
    self.attention = BahdanauAttention(n_units)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

  def initialize_hidden_state(self, batch_size):
    return tf.zeros((batch_size, self.dec_units))
    
  def call(self, x, hidden, enc_output):
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    x, hidden = self.gru(x)
    x = self.fc(x)

    x = tf.reshape(x, (-1, x.shape[2]))

    return x, hidden, attention_weights

