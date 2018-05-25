# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:52:37 2018

@author: athen
"""

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import RMSprop

pstemmer = nltk.PorterStemmer()
wnlemmatizer = nltk.WordNetLemmatizer()

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


df = pd.read_csv('train.tsv', sep='\t')
print(df.head())

#change to binary classification
df.loc[df['Sentiment'] < 2, 'Sentiment'] = 0
df.loc[df['Sentiment'] > 1, 'Sentiment'] = 1
df['Phrase'] = df['Phrase'].str.lower()
df['Phrase'] = df['Phrase'].str.replace('[^\w\s]','')
print(df.head())

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

# Drop stop words. These common words probably won't provide insight into which author wrote each sentence
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

#  We have to: 
#  1. Get the sentences, 
# this takes each sentence from the training file and places it in the list
sents = df['Phrase'].tolist()

# this takes each author id (assigned above) and places them in a list,
# so that we can compare our results to the actual authors
labels = df['Sentiment'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sents)
# turns the text input into numerical arrays
sequences = tokenizer.texts_to_sequences(sents)
print(len(sequences))
print(sequences[0])
##    Get a vector of unique terms here
print('Found %s unique tokens before stopwords removal.' % len(tokenizer.word_index))
print([w for w in tokenizer.word_index.items()][:5])
for w,i in tokenizer.word_index.items():
    wnlemmatizer.lemmatize(w)
word_index = dict([(w,i) for w,i in tokenizer.word_index.items() if w not in stops])
print('Found %s unique tokens after stopwords removal.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
#shuffling indices takes less time than shuffling objects
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# sets the number of the validation samples to 20% of the data (20% is the percentage selected above)
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
y_val[:3]

#  3. Load embeddings
embeddings_index = {}

f = open('C:\\Users\\athen\\Documents\\glove.6B\\glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# This uses ones of several GloVe text files. Using one of the other files may be more effective. The length of the word vectors in the files used is 300. The other options are 50, 100, and 200.

#  4. Create the Embedding matrix for the training set
num_words = min(MAX_NB_WORDS, len(word_index))

# returns an array of the size num_words x EMBEDDING_DIM, filled with 0s
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
unk = []
for word, i in word_index.items():
    #if i >= MAX_NB_WORDS:
    if i >= num_words:
        continue
    # gets the vector for the current word    
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        unk.append(word)
print(len(unk))

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model. Trainable embeddings. Softmax')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='softmax')(embedded_sequences)
x = GlobalMaxPooling1D()(x)
x = Dropout(.3)(x)
x = Dense(128, activation='relu')(x)

preds = Dense(2, activation='softmax')(x)
rms = RMSprop(lr=0.003)
model = Model(sequence_input, preds)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer=rms, #'rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=50,
          epochs=3,
          validation_data=(x_val, y_val))


print('Training model. Trainable embeddings. Two Conv layers')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='softmax')(embedded_sequences)
x = GlobalMaxPooling1D()(x)
x = Conv1D(128, 3, activation='softmax')(embedded_sequences)
x = GlobalMaxPooling1D()(x)
x = Dropout(.3)(x)
x = Dense(128, activation='relu')(x)

preds = Dense(2, activation='softmax')(x)
rms = RMSprop(lr=0.003)
model = Model(sequence_input, preds)
model.compile(loss='mean_squared_logarithmic_error',
              optimizer=rms, #'rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=50,
          epochs=3,
          validation_data=(x_val, y_val))

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# =============================================================================
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# print('Training model. Trainable embeddings. Baseline')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# =============================================================================


# train a 1D convnet with global maxpooling
# =============================================================================
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='softmax')(embedded_sequences)
# x = Conv1D(128, 3, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# =============================================================================



# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
# =============================================================================
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# print('Training model. Trainable embeddings. Linear')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='linear')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# 
# print('Training model. Trainable embeddings. Selu')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='selu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# print('Training model. Trainable embeddings. Linear->Selu->Softmax')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='linear')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='selu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# print('Training model. Trainable embeddings. selu->Linear->softmax')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='selu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='linear')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# =============================================================================

# =============================================================================
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# print('Training model. Trainable embeddings. Softmax. Fewer filters.')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(64, 5, activation='softmax')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# 
# print('Training model. Trainable embeddings. Fewer filters')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(64, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# print('Training model. Trainable embeddings. More filters')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(256, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=50,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# 
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=False)
# 
# print('Training model. No trainable embeddings')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=100,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# print('Training model. Trainable embeddings. No dropout layer')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=100,
#           epochs=3,
#           validation_data=(x_val, y_val))
# 
# # load pre-trained word embeddings into an Embedding layer
# # note that we set trainable = False so as to keep the embeddings fixed
# embedding_layer = Embedding(num_words,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SEQUENCE_LENGTH,
#                             trainable=True)
# 
# print('Training model. Trainable embeddings. More layers')
# 
# # train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(.3)(x)
# 
# preds = Dense(2, activation='softmax')(x)
# rms = RMSprop(lr=0.003)
# model = Model(sequence_input, preds)
# model.compile(loss='mean_squared_logarithmic_error',
#               optimizer=rms, #'rmsprop',
#               metrics=['acc'])
# 
# model.fit(x_train, y_train,
#           batch_size=100,
#           epochs=3,
#           validation_data=(x_val, y_val))
# =============================================================================
