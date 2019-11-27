from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time


path_to_file = ('./lotr.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

print('length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))


char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 256
examples_per_epoch = len(text)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
	print(idx2char[i.numpy()])


sequences = char_dataset.batch(seq_length+1, drop_remainder = True)

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

vocab_size = len(vocab)
embedding_dim = 256

rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
    	tf.keras.layers.Embedding(vocab_size, embedding_dim,
     	batch_input_shape=[batch_size, None]),
    	tf.keras.layers.Dropout(0.2),
     	tf.keras.layers.LSTM(rnn_units,
     	return_sequences=True,
     	stateful=True,
     	recurrent_initializer='glorot_uniform'),
     	tf.keras.layers.Dropout(0.2), 
     	tf.keras.layers.LSTM(rnn_units,
     	return_sequences=True,
     	stateful=True,
     	recurrent_initializer='glorot_uniform'),
     	tf.keras.layers.Dropout(0.2),
     	tf.keras.layers.Dense(vocab_size)])
	return model

model = build_model(vocab_size = len(vocab),
					embedding_dim = embedding_dim,
					rnn_units = rnn_units,
					batch_size = BATCH_SIZE)

model.summary()


optimizer = tf.keras.optimizers.Adam()
def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

model.compile(optimizer = optimizer, loss = loss)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
	filepath = checkpoint_prefix,
	save_weights_only = True)


EPOCHS = 10
history = model.fit(dataset, epochs = EPOCHS, callbacks = [checkpoint_callback])	

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary

def generate_text(model, start_string):
	num_generate = 1000000

	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)


	text_generated = []



	temperature = 1.0


	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)

		predictions = tf.squeeze(predictions, 0)

		predictions = predictions / temperature
		predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()

		input_eval = tf.expand_dims([predicted_id], 0)

		text_generated.append(idx2char[predicted_id])

	return (start_string + ''.join(text_generated))


def run():
	print(generate_text(model, start_string=u"Gandalf struck the Balrog"))	
with open('output.txt', 'w') as f:
	print(generate_text(model, start_string=u"Gandalf struck the Balrog"), file = f)