import pandas as pd 
import numpy as np 
# import argparse as arg 
from argparse import ArgumentParser
import make_word2vec as mw 
import model_preparing as mdl
import preprocessing as pp
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import load_model, Model

embed_dim = 100
max_features = 2000
max_len = 60
batch_size = 32
epochs = 15

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--train_model',default=False)
    parser.add_argument('--word2vec',default=False)
    parser.add_argument('--predict', default=False)

    args = parser.parse_args()

    train = bool(args.train_model)
    predict = bool(args.predict)
    use_word2vec = bool(args.word2vec)

    df_train, df_test = pp.load_dataset()
        #clean word
    dataset_train = df_train[0].apply(lambda x:pp.clean_word(x))
    dataset_test = df_test[0].apply(lambda x:pp.clean_word(x))
    #delete punc
    dataset_train = dataset_train.apply(lambda x: pp.clean_punct(x))
    dataset_test = dataset_test.apply(lambda x: pp.clean_punct(x))
    #normalization
    dataset_train = dataset_train.apply(lambda x: pp.normalization(x))
    dataset_test = dataset_test.apply(lambda x: pp.normalization(x))

    # #get word-index
    tokenizer = Tokenizer(num_words=2000,oov_token='<OOV>')
    tokenizer.fit_on_texts(list(dataset_train))
    word_index = tokenizer.word_index

    if train:
        
        #preparing sequences
        #pad sequences
        print(dataset_train)
        X_train = tokenizer.texts_to_sequences(dataset_train)
        X_train = sequence.pad_sequences(X_train, padding='post', truncating='post',maxlen=60)
        X_test = tokenizer.texts_to_sequences(dataset_test)
        X_test = sequence.pad_sequences(X_test, padding='post', truncating='post',maxlen=60)

        y_train = df_train[1].apply(lambda x : 1 if x =='positive' else 0)
        y_test =df_test[1].apply(lambda x : 1 if x =='positive' else 0)

        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values

        #preparing model
        words = len(word_index) + 1
        if use_word2vec:
            # #make word2vec
            # data = dataset_train.apply(lambda x : x.split())
            # mw.making_model(data)

            # #preparing word embedding
            embedding_matrix = mw.preparing_embedding(word_index)
            model = mdl.build_model_wv(words,embed_dim,embedding_matrix)
            filepath = 'model_wv'
        else:
            model = mdl.build_model(words,embed_dim)
            filepath = 'model'

        #Training
        print('Train...')
        
        best_model = mdl.train(X_train,y_train,X_test,y_test,batch_size, epochs, model,filepath)
        
        #evaluasi
        label = mdl.label_normal(df_test[1])
        score,acc = mdl.evaluasi(X_test, y_test, label, batch_size,best_model)

    elif predict:
        text = input("Text: ")
        text = pp.clean_word(text)
        text = pp.clean_punct(text)
        X_text = tokenizer.texts_to_sequences([text])
        X_text = sequence.pad_sequences(X_text, padding='post', truncating='post',maxlen=max_len)
        if use_word2vec:
            model_name = "model_wv_train_model.h5"
        else:
            model_name = "model_train_model.h5"
        label = mdl.prediction(X_text,dataset_train,max_len,model_name)
        print("Label: ", label)