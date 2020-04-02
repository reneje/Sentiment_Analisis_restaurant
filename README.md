# Sentiment Analysis Restaurant berbahasa Indonesia

This project for NLP Task Prosa.ai

install all requirement
```bashcd 
pip install -r requirements.txt
```

main program is main.py

if you want to train model
```bash
python main.py --train_model=True
```
if you want to train model using word2vec
word2vec is used, i built from dataset
```bash
python main.py --train_model=True --word2vec=True
```

if you want to predict a text
```bash
python main.py --predict=True
```
or if you want to predict a text with model using word2vec
```bash
python main.py --predict=True --word2vec=True
```

and then you can input a text/sentence
```bash
Text : ditempat ini satenya enak hanya saja bumbunya ngga masuk menurut saya, bumbunya kurang enak, overalll jadi biasa saja
```
result:
```bash
label: negative
```

----------
Using other word2vec.
you can using other word2vec
but, i don't guarantee the program will run smoothly
but you can built your word2vec using ur data collection. you can find the program in make_word2vec.py
```bash
from gensim.models import Word2Vec
#your data collection one row fill of word/token
#example:
# data_collection = [['aku','sudah','makan'],['dia','bermain','bola']]
sentences = list(data_collection)
# train model
model = Word2Vec(sentences,min_count=2)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model[model.wv.vocab])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
```

