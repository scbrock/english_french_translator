import os
import os.path
from os import path
import sys

import collections
import numpy as np

import pickle
import re

#import project_tests as tests
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.model_selection import train_test_split



class EFTranslator:

    def __init__(self, model_name='models/final_model.h5'):
        if model_name is not None:
            self.model = load_model(model_name)
        else:
            self.model = None
        self.wv = None
        self.english_tokenizer = None
        self.french_tokenizer = None
        self.english_sentences = None
        self.french_sentences = None

    # model_name = 'models/final_model.h5'

    # model = load_model(model_name)


    # load the data
    def load_data(self, path):
        """
        Load dataset of English and French words
        """
        input_file = os.path.join(path)
        with open(input_file, "r") as f:
            data = f.read()

        return data.split('\n')

    def load_en(self, path='data/small_vocab_en.txt'):
        self.english_sentences = self.load_data(path)
        self.english_sentences = [s.replace('.','').replace(',','').replace('?','').strip().lower() for s in self.english_sentences]

    def load_fr(self, path='data/small_vocab_fr.txt'):
        self.french_sentences = self.load_data('data/small_vocab_fr.txt')
        self.french_sentences = [s.replace('.','').replace(',','').replace('?','').replace('-',' ').lower() for s in self.french_sentences]

    def get_counters(self):
        self.english_words_counter = collections.Counter([word for sentence in self.english_sentences for word in sentence.split()])
        self.french_words_counter = collections.Counter([word for sentence in self.french_sentences for word in sentence.split()])
    # print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
    # print('{} unique English words.'.format(len(english_words_counter)))
    # print('10 Most common words in the English dataset:')
    # print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    # print()
    # print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
    # print('{} unique French words.'.format(len(french_words_counter)))
    # print('10 Most common words in the French dataset:')
    # print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


    def tokenize(self, x):
        '''
        Create a tokenizer that can process English words and convert them into numbers
        '''
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


    def pad(self, x, length=None):
        '''
        add padding so that input all has same length - padding is added to the end
        '''
        if length is None:
            length = max([len(sentence) for sentence in x])
        return pad_sequences(x, maxlen = length, padding = 'post')
    #tests.test_pad(pad)
    # Pad Tokenized output
    # test_pad = pad(text_tokenized)
    # for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    #     print('Sequence {} in x'.format(sample_i + 1))
    #     print('  Input:  {}'.format(np.array(token_sent)))
    #     print('  Output: {}'.format(pad_sent))


    def preprocess(self, x, y):
        '''
        convert x (English) into numbers representing them, not one hot
        convert y (French) into numbers representing the words, not one hot
        '''
        preprocess_x, x_tk = self.tokenize(x)
        preprocess_y, y_tk = self.tokenize(y)
        preprocess_x = self.pad(preprocess_x)
        preprocess_y = self.pad(preprocess_y)
        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
        return preprocess_x, preprocess_y, x_tk, y_tk
    
    # self.preproc_english_sentences, self.preproc_french_sentences, self.english_tokenizer, self.french_tokenizer =\
    #     self.preprocess(english_sentences, french_sentences)

    #print('preprocess_french_sentences, y:', preproc_french_sentences[:10])
    #print('max french word:', np.max(preproc_french_sentences))
    # print('english tokenizer word index', english_tokenizer.word_index)
    # print('\n')

    # print('counter items:',english_words_counter.items())

    # print('\n')

    def check_balance(self, english_words_counter,english_tokenizer,french_words_counter,french_tokenizer):
        '''
        determine whether there are missing characters in the english/french tokenizers
        '''

        a = set([t[0] for t in list(english_words_counter.items())])
        b = set(list(english_tokenizer.word_index.keys()))

        fa = set([t[0] for t in list(french_words_counter.items())])
        fb = set(list(french_tokenizer.word_index.keys()))


        # print('list(english_counter.items()):',[t[0] for t in list(english_words_counter.items())])

        #print('keys:', list(english_tokenizer.word_index.keys()))

        print('Checking word set differences...')
        print('english set difference:', a.difference(b))
        # english set difference: {'favorite.', 'strawberry.', 'lime.', 'peaches.', '?', 'fruit.', '.', 'lemon.', 'pears.', 
        # 'grapes.', 'bananas.', 'orange.', 'loved.', 'lemons.', 'pear.', 'grapefruit.', 'grape.', 'apple.', 'oranges.', 'peach.', 'apples.', 'mangoes.', 'liked.', 'banana.', 'limes.', 'mango.', ',', 'strawberries.'}
        print('french set difference:', fa.difference(fb))  
        # french set difference: {'-elle', 'es-tu', 'aiment-ils', 'est-ce', 'etats-unis', '-', 'préféré.', 'êtes-vous', 'as-tu', 'états-unis', '?', '-ce', 'aimé.', 'États-unis', '.', ',', '-il', '-ils'}
        #print('counter difference tokenizer:', set(list(english_words_counter.items())).difference(set(english_tokenizer.word_index.keys())))
        return

    # self.check_balance(english_words_counter,english_tokenizer,french_words_counter,french_tokenizer)

    # max_english_sequence_length = self.preproc_english_sentences.shape[1]
    # max_french_sequence_length = self.preproc_french_sentences.shape[1]
    # english_vocab_size = len(self.english_tokenizer.word_index)
    # french_vocab_size = len(self.french_tokenizer.word_index)
    # print('Data Preprocessed')
    # print("Max English sentence length:", max_english_sequence_length)
    # print("Max French sentence length:", max_french_sequence_length)
    # print("English vocabulary size:", english_vocab_size)
    # print("French vocabulary size:", french_vocab_size)


    def logits_to_text(self, logits, tokenizer):
        '''
        converting from prediction back to words
        '''
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])
    #print('`logits_to_text` function loaded.')


    # build tokenizer for model
    # save tokenizer??

    def predict_sentences(self, s):
        x_tk = self.english_tokenizer
        y_tk = self.french_tokenizer
        model = self.model
        
        y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
        y_id_to_word[0] = '<PAD>'


        sentences = []
        for sent in s:
            sent = sent.lower()
            error = False
            # verify each word is in the dictionary
            for w in sent.split():
                if w not in x_tk.word_index:
                    print(w,'not in the English training set')
                    error = True
                    break
            if error:
                break

            # y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
            # y_id_to_word[0] = '<PAD>'
            # sentence = 'he saw a old yellow truck'
            # sentence = [x_tk.word_index[word] for word in sentence.split()]
            # sentence = pad_sequences([sentence], maxlen=max_english_sequence_length, padding='post')
            # sentences = np.array([sentence[0], x[0]])  # x[0] is a data point from the input

            sentence = [x_tk.word_index[word] for word in sent.split()]
            #print('sent 1',sentence)
            sentence = pad_sequences([sentence], maxlen=self.max_french_sequence_length, padding='post')
            #print('sent 2', sentence)
            sentences.append(sentence[0].tolist())

        #print('sentences')
        print('sentences:',sentences)

        predictions = model.predict(np.array(sentences), steps=len(sentences))
        #print('predictions shape')
        #print(predictions.shape)

        print('generating predictions')
        #for i, p in enumerate(predictions[:len(predictions)-1]):
        for i in range(len(s)):
            p = predictions[i]
            print('original')
            print(s[i])
            print('prediction')
            print(' '.join([y_id_to_word[np.argmax(w)] for w in p]))


    # What happens if we try to translate a word that we haven't seen before?
    # error if word not in dictionary


    #self.predict_sentences(to_translate)

    def run_preprocess(self):
        self.load_en()
        self.load_fr()
        self.get_counters()
        self.preproc_english_sentences, self.preproc_french_sentences, self.english_tokenizer, self.french_tokenizer =\
            self.preprocess(self.english_sentences, self.french_sentences)

        # try saving and loading tokenizers
        # saving
        with open('entokenizer.pickle', 'wb') as handle:
            pickle.dump(self.english_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('frtokenizer.pickle', 'wb') as handle:
            pickle.dump(self.french_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # loading
        with open('entokenizer.pickle', 'rb') as handle:
            self.english_tokenizer = pickle.load(handle)
        
        with open('frtokenizer.pickle', 'rb') as handle:
            self.french_tokenizer = pickle.load(handle)
        
        self.check_balance(self.english_words_counter,self.english_tokenizer,self.french_words_counter,self.french_tokenizer)

        self.max_english_sequence_length = self.preproc_english_sentences.shape[1]
        self.max_french_sequence_length = self.preproc_french_sentences.shape[1]
        self.english_vocab_size = len(self.english_tokenizer.word_index)
        self.french_vocab_size = len(self.french_tokenizer.word_index)

        # create train and test sets
        self.ppe_train, self.ppe_test, self.ppf_train, self.ppf_test = \
            train_test_split(self.preproc_english_sentences, self.preproc_french_sentences, test_size=0.2, random_state=23)

        self.preprocess_wv()

    
    def model_final(self, input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
        model = Sequential()
        model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))

        model.add(Bidirectional(LSTM(256,return_sequences=True)))
        model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
        learning_rate = 0.005
        
        model.compile(loss = sparse_categorical_crossentropy, 
                    optimizer = Adam(learning_rate), 
                    metrics = ['accuracy'])
        #print(model.summary())
        
        return model

    def create_model(self):
        padded_input = self.pad(self.ppe_train, self.ppf_train.shape[1])
        #padded_input = self.pad2(self.prepro_eng)
        #print('padded_input.shape',padded_input.shape)
        #print('top 5', padded_input[:5])



        self.model = self.model_final(padded_input.shape,
                    self.preproc_french_sentences.shape[1],
                    len(self.english_tokenizer.word_index)+1,
                    len(self.french_tokenizer.word_index)+1)
        self.model.fit(padded_input, self.ppf_train, batch_size = 1024, epochs = 10, validation_split = 0.2)
        self.model.save('models/final_model.h5')
    
    def test_model(self):
        '''
        return the accuracy of the model
        '''
        padded_input = self.pad(self.ppe_test, self.ppf_test.shape[1])
        self.pred = self.model.predict(padded_input)
        self.pred = [[[np.argmax(w)] for w in s] for s in self.pred]
        #print('test pred first 10:',self.pred[:10])
        #print('target first 10:',self.ppf_test[:10])

        comparison_matrix = np.equal(self.pred, self.ppf_test)
        comp_sent = [np.alltrue(s) for s in comparison_matrix]
        print('sentence accuracy on test set:', np.sum(comp_sent)/len(comp_sent))
        print('word accuracy on test set:', np.sum(comparison_matrix)/np.product(comparison_matrix.shape))

    
    def load_wv(self,file = 'vw_enfr.wordvecs'):
        self.wv = gensim.models.KeyedVectors.load(file)

    # I need to change the preprocess in order to manually do the word to vector part

    def preprocess_wv(self):
        '''
        Assume english and french sentences already loaded
        Also assume that load wv has been called
        '''
        if self.wv is None:
            self.load_wv()

        print('keys:',list(self.wv.vocab))
        
        prepro_eng = list()
        i = 0
        for s in self.english_sentences:
            i += 1
            if i == 10:
                break
            for w in word_tokenize(s.replace('  ',' ')):
                
                try:
                    wemb = self.wv[w]
                    prepro_eng.append(wemb)
                except Exception as e:
                    print('sentence:',s)
                    print('word:',w)

        self.prepro_eng = [[self.wv[w] for w in word_tokenize(s.replace('  ',' '))] for s in self.english_sentences]
        self.prepro_fr = self.preproc_french_sentences
    
    def pad2(self, data, length=None):
        '''
        pad data that has already been transformed into vector embeddings
        '''

        if length is None:
            length = max([len(v) for v in data])  # get length of longest sentence vector

        word_dim = len(data[0][0])
        print('len of each word:',word_dim)
        print('maxlen:',length)

        # update each sentence, add vectors of zeros to end of each sentence
        # adjusted = [s + [[0 for _ in range(word_dim)] for _ in range(length-len(s))] for s in data]
        # return np.array(adjusted)





def main():
    print('running main')


    to_translate = [
        #'I want to go with you',
        'I like you',
        # Hi how are you',
        #'what is your name',
        'the united states are quiet during spring',
        'the united states is usually chilly during july and it is usually freezing in november',
        'winter is cold'
    ]

    # create model instance
    model = EFTranslator(model_name=None)
    model.run_preprocess()
    if model.model is None:
        model.create_model()
    model.test_model()
    print('Hi there, I am ETFTranslator Bot. I can translate sentences from English to French for you from the following vocabulary.')
    print('vocab:')
    print(sorted(list(model.english_tokenizer.word_index.keys())))
    #model.predict_sentences(model, to_translate, english_tokenizer, french_tokenizer)

    while True:
        try:
            x = input('enter sentence to translate: ')
        except Exception as e:
            break
        print('input:',x)
        line = x.replace('.','').replace('?','').replace(',','').lower()
        model.predict_sentences([x])
        print('\n')
    print('\n')


if __name__ == '__main__':
    main()




