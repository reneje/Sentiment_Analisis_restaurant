#how to making word2vec from data
import numpy as np
from gensim.models import Word2Vec

def making_model(data):
    sentences = list(data)
    # train model
    model = Word2Vec(sentences,min_count=2)
    # summarize the loaded model
    # print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    # print(model[model.wv.vocab])
    # save model
    model.save('model.bin')
    # load model
    new_model = Word2Vec.load('model.bin')
    # print(new_model)

def preparing_embedding(word_index):
    model = Word2Vec.load('model 2.bin')
    words_in_word2vec = model.wv.vocab
    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words, 100))
    for word, i in word_index.items():
        if word in words_in_word2vec:
            embedding_matrix[i] = model[word]

    return embedding_matrix