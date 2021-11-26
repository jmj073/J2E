from konlpy.tag import Mecab
import os
from models import *

MECAB_DICT_PATH = "C:/mecab/mecab-ko-dic"
TOKENIZER_PATH = 'resource/word_table(keras_tokenizer)'
INPUT_TOKENIZER_PATH = os.path.join(TOKENIZER_PATH, 'J2E_input_tokenizer.pickle')
OUTPUT_TOKENIZER_PATH = os.path.join(TOKENIZER_PATH, 'J2E_output_tokenizer.pickle')
MODEL_CHACKPOINT_PATH = './resource/translator'

MODEL_INFO = {'input_vocab_size': 28804, 'output_vocab_size': 8151, 'units': 1024, 'embedding_dim': 256}

def make_translator(encoder, decoder, input_tokenizer, output_tokenizer):
    mecab = Mecab(MECAB_DICT_PATH)
    def translate(text, maxlen=100):
        results = ""

        text = ['<sos>'] + mecab.morphs(text) + ['<eos>']
        text = input_tokenizer.texts_to_sequences(text)
        text = np.array([[i[0] for i in text if i]])

        enc_hidden = encoder.initialize_hidden_state(1)
        enc_output, dec_hidden = encoder(text, enc_hidden)

        dec_token = tf.reshape(output_tokenizer.word_index['<sos>'], (1, 1))
        eos_token = output_tokenizer.word_index['<eos>']

        for i in range(maxlen):
            dec_token, dec_hidden, _ = decoder(dec_token, dec_hidden, enc_output)

            dec_token = tf.argmax(dec_token[0]).numpy()
            if dec_token == eos_token:
                break

            results += ' ' + output_tokenizer.index_word[dec_token]

            dec_token = tf.expand_dims([dec_token], 0)

        return results
    
    return translate

def load_tokenizers():
    import pickle

    with open(INPUT_TOKENIZER_PATH, 'rb') as handle:
        input_tokenizer = pickle.load(handle)

    with open(OUTPUT_TOKENIZER_PATH, 'rb') as handle:
        output_tokenizer = pickle.load(handle)

    return input_tokenizer, output_tokenizer

def load_enocder_decoder():
    encoder = Encoder(MODEL_INFO['units'], MODEL_INFO['input_vocab_size'], MODEL_INFO['embedding_dim'])
    decoder = Decoder(MODEL_INFO['units'], MODEL_INFO['output_vocab_size'], MODEL_INFO['embedding_dim'])

    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(MODEL_CHACKPOINT_PATH)).expect_partial()

    return encoder, decoder

def load_translator():
    return make_translator(*load_enocder_decoder(), *load_tokenizers())