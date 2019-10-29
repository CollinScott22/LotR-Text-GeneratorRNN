from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time



path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path, 'rb').read().decode(encoding='utf-8')
print('length of text:: {} characters'.format(len(text)))

print(text[:250])


vocabulary = sorted(set(text))
print( '{} unique characters'.format(len(vocab)))

char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)


text_as_int = np.array([char2idx[c] for c in text])


sequence_length = 200
epoch_examples = len(text * 2)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


for i in char_dataset.take(5)
	print(idx2char[i.numpy()])

sequences = char_dataset.batch(sequence_length+1, drop_remainder=True)
for item in sequences.take(5):
	print(repr(''.join(id2char[item.numpy()])))

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[:-1]
	return input_text, target_text

dataset = sequences.map(split_input_target)


BATCH_SIZE = 43



BUFFER_SIZE = 12500

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset



vocab_size = len(vocab)

embedding_dimensions = 256


rnn_units = 2048