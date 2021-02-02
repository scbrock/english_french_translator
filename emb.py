'''
emb.py is a file where I create and save a word2vec model
'''

import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import os
from os import path
import sklearn
#nltk.download('punkt')


def create_wordvec(output_file):
    '''
    create the word to vec mapping dictionary and save to output_file

    args:
        output_file - string, output file name

    returns:
        None
    '''

    sample = open('data/small_vocab_en.txt','r')
    s = sample.read()
    #print(s[:10])
    #f = s.replace('\n',' ')

    def clean_punctuation(sentences):
        '''
        remove unnecessary punctuation
        '''
        return [s.replace('?','').replace('.',"").replace(',','').replace('-', ' ').lower() for s in sentences]


    s_clean = clean_punctuation(s.split('\n'))

    tmp = s[:10]

    #s2 = s.replace('\n','')
    #print(s2)
    s3 = sent_tokenize(s)

    #print(s3[:10])
    #print(word_tokenize(s3[0])[:10])

    data = [word_tokenize(s) for s in s_clean]

    model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                                window = 5, sg = 1)
    # sg = 1 means skip-gram 0, means CBOW

    word_vectors = model2.wv
    word_vectors.save(output_file)


    print("Cosine similarity between 'peach' " +
            "and 'fruit' - Skip Gram : ", 
        model2.similarity('peach', 'fruit')) 

    print("Cosine similarity between 'winter' " +
                "and 'spring' - Skip Gram : ", 
        model2.similarity('winter', 'spring')) 

file = 'vw_enfr.wordvecs'

if path.exists(file):
    print('loading word2vec mappings...')
    wv = gensim.models.KeyedVectors.load(file)
else:
    print('saving word2vec mappings...')
    create_wordvec(file)
    wv = gensim.models.KeyedVectors.load(file)
    print('loading word2vec mappings...')

print("Cosine similarity between 'peach' " +
        "and 'fruit' - Skip Gram : ", 
    sklearn.metrics.pairwise.cosine_similarity([wv['peach'],wv['fruit']])) 

print("Cosine similarity between 'winter' " +
            "and 'spring' - Skip Gram : ", 
    sklearn.metrics.pairwise.cosine_similarity([wv['winter'],wv['spring']])) 


