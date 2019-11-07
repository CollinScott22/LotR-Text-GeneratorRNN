
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


text = (open("lotr.txt").read())
text = text.lower()


characters = sorted(list(set(text)))

n_to_char ={n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}

vocab_size = len(characters)
print('number of unique characters: ', vocab_size)
print(characters)


x = []
y = []
length = len(text)
seq_len = 100


for i in range(0, length - seq_len, 1):
	sequence = text[i:i + seq_len]
	label = text[i + seq_len]
	x.append([char_to_n[char] for char in sequence])
	y.append(char_to_n[label])

x_mod = np.reshape(x, (len(x), seq_len, 1))
x_mod = x_mod / float(len(characters))
y_mod = np_utils.to_categorical(y)


model = Sequential()
model.add(LSTM(700, input_shape=(x_mod.shape[1],x_mod.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(700, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(y_mod.shape[1], activation='softmax'))
