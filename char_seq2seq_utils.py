#Character seq-to-seq model
from __future__ import print_function
from keras.preprocessing.text import text_to_word_sequence,Tokenizer
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding,Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
import nltk
from nltk import FreqDist
import numpy as np
import os
import datetime
from keras.layers import Bidirectional


def process_labels(y_data,X_data):
	x=X_data.split("\n")
	y=y_data.split("\n")
	labels=[]
	for i in range(len(x)):
		c=x[i].split()
		d=y[i].split()
		ans=""
		for j in range(len(c)):
			e=list(c[j])
			for k in range(len(e)):
				ans=ans+"NONE"+" "
			ans=ans+d[j]+" "
		labels.append(ans)
	final="\n".join(labels)
	return final

def load_data(source, dist, max_len, vocab_size):

    # Reading raw text from source and destination files
    f = open(source, 'r')
    X_data = f.read()
    f.close()
    f = open(dist, 'r')
    y_data = f.read()
    f.close()

    # Splitting raw text into array of sequences
    X = [list(x+' ') for x in X_data.split('\n') if len(x) > 0 and len(x) <= max_len]
    y_data=process_labels(y_data,X_data)
    y = [text_to_word_sequence(y) for y in y_data.split('\n') if len(y) > 0 and len(y) <= max_len]
    # Creating the vocabulary set with the most common words
    dist = FreqDist(np.hstack(X))
    X_vocab = dist.most_common(vocab_size-1)
    dist = FreqDist(np.hstack(y))
    y_vocab = dist.most_common(vocab_size-1)

    # Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
    X_ix_to_word = [word[0] for word in X_vocab]
    # Adding the word "ZERO" to the beginning of the array
    X_ix_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    X_ix_to_word.append('UNK')

    # Creating the word-to-index dictionary from the array created above
    X_word_to_ix = {word:ix for ix, word in enumerate(X_ix_to_word)}

    # Converting each word to its index value
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']

    y_ix_to_word = [word[0] for word in y_vocab]
    y_ix_to_word.insert(0, 'ZERO')
    y_ix_to_word.append('UNK')
    y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}
    for i, sentence in enumerate(y):
        for j, word in enumerate(sentence):
            if word in y_word_to_ix:
                y[i][j] = y_word_to_ix[word]
            else:
                y[i][j] = y_word_to_ix['UNK']
    return (X, len(X_vocab)+2, X_word_to_ix, X_ix_to_word, y, len(y_vocab)+2, y_word_to_ix, y_ix_to_word)

def load_test_data(source, X_word_to_ix, max_len):
    f = open(source, 'r')
    X_data = f.read()
    f.close()

    X = [list(x+' ') for x in X_data.split('\n') if len(x) > 0 and len(x) <= max_len]
    
    for i, sentence in enumerate(X):
        for j, word in enumerate(sentence):
            if word in X_word_to_ix:
                X[i][j] = X_word_to_ix[word]
            else:
                X[i][j] = X_word_to_ix['UNK']
    return X


def create_model(X_vocab_len, X_max_len, y_vocab_len, y_max_len, hidden_size, num_layers):
    model = Sequential()
    model.add(Embedding(X_vocab_len, 1000, input_length=X_max_len, mask_zero=True))  
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(hidden_size,return_sequences=True, recurrent_dropout=0.1)))  
    model.add(Bidirectional(LSTM(hidden_size,return_sequences=True, recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(y_vocab_len,activation="softmax")))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model_json = model.to_json()
    with open("model.json", "w") as json_file: 
                     json_file.write(model_json)
    return model


def process_data(word_sentences, max_len, word_to_ix):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def process_output(sequences,names):
    f = open(names, 'r')
    X_data = f.read()
    f.close()

    X = [list(x+' ') for x in X_data.split('\n') if len(x) > 0]
    
    final=[]
    for i in range(len(X)):
        id=findOccurrences(X[i],' ')
        seq=sequences[i].split()
        ans=''
        for j in range(len(id)):
            ans=ans+seq[id[j]]+' '
        final.append(ans)
    return final

def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    a=checkpoint_file[np.argmax(modified_time)]
    print(a)
    return a
