import collections
import numpy as np
#import project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential

import os
import os.path
from os import path



def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

english_sentences = load_data('data/small_vocab_en.txt')
french_sentences = load_data('data/small_vocab_fr.txt')
print('Dataset Loaded')

# remove '.', ',', '?' from english
# remove '-' -> ' ', '.', ',', '?'
english_sentences = [s.replace('.','').replace(',','').replace('?','').lower() for s in english_sentences]
french_sentences = [s.replace('.','').replace(',','').replace('?','').replace('-',' ').lower() for s in french_sentences]
print('removed punctuation')

for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))

# find all the strings that have a period attached to the last word


english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
# print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
# print('{} unique English words.'.format(len(english_words_counter)))
# print('10 Most common words in the English dataset:')
# print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
# print()
# print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
# print('{} unique French words.'.format(len(french_words_counter)))
# print('10 Most common words in the French dataset:')
# print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

# text_sentences = [
#     'The quick brown fox jumps over the lazy dog .',
#     'By Jove , my quick study of lexicography won a prize .',
#     'This is a short sentence .']

# text_tokenized, text_tokenizer = tokenize(text_sentences)
# print(text_tokenizer.word_index)
# print()
# for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
#     print('Sequence {} in x'.format(sample_i + 1))
#     print('  Input:  {}'.format(sent))
#     print('  Output: {}'.format(token_sent))


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

# # Pad Tokenized output
# test_pad = pad(text_tokenized)
# for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
#     print('Sequence {} in x'.format(sample_i + 1))
#     print('  Input:  {}'.format(np.array(token_sent)))
#     print('  Output: {}'.format(pad_sent))


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =\
    preprocess(english_sentences, french_sentences)

#print('preprocess_french_sentences, y:', preproc_french_sentences[:10])
print('max french word:', np.max(preproc_french_sentences))
# print('english tokenizer word index', english_tokenizer.word_index)
# print('\n')

# print('counter items:',english_words_counter.items())

# print('\n')

a = set([t[0] for t in list(english_words_counter.items())])
b = set(list(english_tokenizer.word_index.keys()))

fa = set([t[0] for t in list(french_words_counter.items())])
fb = set(list(french_tokenizer.word_index.keys()))


# print('list(english_counter.items()):',[t[0] for t in list(english_words_counter.items())])

print('keys:', list(english_tokenizer.word_index.keys()))

print('english set difference:', a.difference(b))
# english set difference: {'favorite.', 'strawberry.', 'lime.', 'peaches.', '?', 'fruit.', '.', 'lemon.', 'pears.', 
# 'grapes.', 'bananas.', 'orange.', 'loved.', 'lemons.', 'pear.', 'grapefruit.', 'grape.', 'apple.', 'oranges.', 'peach.', 'apples.', 'mangoes.', 'liked.', 'banana.', 'limes.', 'mango.', ',', 'strawberries.'}
print('french set difference:', fa.difference(fb))  
# french set difference: {'-elle', 'es-tu', 'aiment-ils', 'est-ce', 'etats-unis', '-', 'préféré.', 'êtes-vous', 'as-tu', 'états-unis', '?', '-ce', 'aimé.', 'États-unis', '.', ',', '-il', '-ils'}

#print('counter difference tokenizer:', set(list(english_words_counter.items())).difference(set(english_tokenizer.word_index.keys())))
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
print('`logits_to_text` function loaded.')


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)  # what does time-distributed mean? sequence?
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model
#tests.test_simple_model(simple_model)
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size+1)

#simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=1, validation_split=0.2)

# save model
# print('saving model...')
# simple_rnn_model.save('models/simple_rnn_model.h5')
# print('finished saving.')




# # Print prediction(s)
# print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))


# # load model
# print('loading model...')
# reconstructed_model = load_model("models/simple_rnn_model.h5")
# print('finished loading.')

# print('testing model')
# np.testing.assert_allclose(
#     simple_rnn_model.predict(tmp_x), reconstructed_model.predict(tmp_x)
# )


print('vector embedding model...')


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation="tanh")
    
    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1]) 
    logits = TimeDistributed(Dense(french_vocab_size, activation="softmax"))
    
    model = Sequential()
    #em can only be used in first layer --> Keras Documentation

    # add embedding layer
    model.add(embedding)

    # add rnn layer
    model.add(rnn)

    # add time distributed layer
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model


tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))


if path.exists('models/embed_model.h5'):
    print('embed exists and loading...')
    embeded_model = load_model('models/embed_model.h5')
else:
    print('embed does not exist. Creating and saving...')
    embeded_model = embed_model(
        tmp_x.shape,
        max_french_sequence_length,
        english_vocab_size,
        french_vocab_size)
    embeded_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)
    embeded_model.save('models/embed_model.h5')


print(logits_to_text(embeded_model.predict(tmp_x[:1])[0], french_tokenizer))


print('bidirectional model...')


def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
   
    learning_rate = 1e-3
    # create the model
    model = Sequential() 

    # add a bidrectional GRU (both ways)
    # review what bidirectional framework looks like
    model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1), 
                           input_shape = input_shape[1:]))
    
    # what is a time distributed dense layer?
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))  # mapping outputs by taking the softmax

    # compile the model togther
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])

    print('input shape',input_shape)
    print('bidirectional model summary')
    print(model.summary())
    return model
#tests.test_bd_model(bd_model)
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))


if path.exists('models/bidi_model.h5') and False:
    print('bidi model exists. Loading...')
    bidi_model = load_model('models/bidi_model.h5')
else:
    print('bidi model does not exist. Creating and saving...')
    bidi_model = bd_model(
        tmp_x.shape,
        preproc_french_sentences.shape[1],
        len(english_tokenizer.word_index)+1,
        len(french_tokenizer.word_index)+1)

    bidi_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
    bidi_model.save('models/bidi_model.h5')

# fit the model

# Print prediction(s)
print(logits_to_text(bidi_model.predict(tmp_x[:1])[0], french_tokenizer))



#
print('running encoder-decoder model...')

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences = True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    return model
#tests.test_encdec_model(encdec_model)
tmp_x = pad(preproc_english_sentences)
tmp_x = tmp_x.reshape((-1, preproc_english_sentences.shape[1], 1))

if path.exists('models/encdc_model.h5'):
    print('loading encdc_model...')
    encodeco_model = load_model('models/encdc_model.h5')
else:
    print('creating and saving model...')
    encodeco_model = encdec_model(
        tmp_x.shape,
        preproc_french_sentences.shape[1],
        len(english_tokenizer.word_index)+1,
        len(french_tokenizer.word_index)+1)
    encodeco_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
    encodeco_model.save('models/encdc_model.h5')

print(logits_to_text(encodeco_model.predict(tmp_x[:1])[0], french_tokenizer))



# sequential model

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model


def final_predictions(x, y, x_tk, y_tk):
    tmp_X = pad(preproc_english_sentences)

    if path.exists('models/final_model.h5'):
        model = load_model('models/final_model.h5')
    else:
        model = model_final(tmp_X.shape,
                    preproc_french_sentences.shape[1],
                    len(english_tokenizer.word_index)+1,
                    len(french_tokenizer.word_index)+1)
        
        model.fit(tmp_X, preproc_french_sentences, batch_size = 1024, epochs = 17, validation_split = 0.2)
        model.save('models/final_model.h5')

    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    sentence = 'he saw a old yellow truck'
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=max_english_sequence_length, padding='post')
    sentences = np.array([sentence[0], x[0]])  # x[0] is a data point from the input
    print('predicting sentences:',sentences)

    predictions = model.predict(sentences, steps=len(sentences))
    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))
final_predictions(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer)


test1 = 'je veux jouer avec toi'
test1e = 'I want to play with you'

