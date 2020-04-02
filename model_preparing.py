#model deep learning
import pandas as pd
import numpy as np
# from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report, accuracy_score,confusion_matrix
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.models import load_model, Model

# df_train = pd.read_csv("train_data_restaurant.tsv", delimiter='\t', header=None)
# df_train= shuffle(df_train)
# df_test = pd.read_csv("test_data_restaurant.tsv", delimiter='\t', header=None)
# df_test=shuffle(df_test)


def get_word_index(list_dataset):
    tokenizer = Tokenizer(num_words=2000,oov_token='<OOV>')
    tokenizer.fit_on_texts(list(list_dataset))
    word_index = tokenizer.word_index

    return word_index

def sequences_data(data,data_train,max_len):
    tokenizer = Tokenizer(num_words=2000,oov_token='<OOV>')
    tokenizer.fit_on_texts(list(data_train))
    x = tokenizer.texts_to_sequences(data)
    x = sequence.pad_sequences(x, padding='post',truncating='post',maxlen=max_len)

    return x

def label_proses(data):
    y = data.apply(lambda x : 1 if x =='positive' else 0)
    return pd.get_dummies(y).values

def label_normal(data):
    return list(data.apply(lambda x : 1 if x =='positive' else 0))

def build_model(words,embed_dim):
    models = Sequential()
    models.add(Embedding(words,embed_dim, mask_zero=True))
    models.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))  # 100 memory
    models.add(Dense(units=2,activation='softmax'))

    # try using different optimizers and different optimizer configs
    models.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    models.summary()

    return models

def build_model_wv(words,embed_dim,embedding_matrix):
    models = Sequential()
    models.add(Embedding(words,embed_dim, weights = [embedding_matrix], mask_zero=True))
    models.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))  # 100 memory
    models.add(Dense(units=2,activation='softmax'))

    # try using different optimizers and different optimizer configs
    models.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    models.summary()

    return models

def train(X_train,y_train,X_test,y_test,batch_size, epochs, models,filepath):

    best_train = filepath+'_train_model.h5'
    mcp_save = ModelCheckpoint(best_train, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    models.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=15,
            validation_data=[X_test,y_test],
            callbacks=[mcp_save,reduce_lr_loss])

    return best_train
    
def evaluasi(X_test, y_test, label, batch_size,best_train):
    model_prediction = load_model(best_train)
    score, acc = model_prediction.evaluate(X_test, y_test,
                            batch_size=batch_size)
    
    #analisis
    pred = model_prediction.predict(X_test)
    y_test_1 = label
    preds = np.array([np.argmax(pr) for pr in pred])
    correct = np.sum(preds==y_test_1)
    print ("Correctly Predicted : ", correct,"/",len(y_test_1))
    print ("Accuracy : ", correct*100.0/len(y_test_1))
    print(confusion_matrix(preds, y_test_1))
    print(classification_report(preds, y_test_1))
    return score, acc

def prediction(X_text,df_train,max_len,model_name):
    #data dalam bentuk raw
    model_prediction = load_model(model_name)
    pred = model_prediction.predict([X_text])
    preds = np.array([np.argmax(pr) for pr in pred])
    # print(preds)
    if preds == 1:
        label = "positive"
    else:
        label = "negative"

    return label