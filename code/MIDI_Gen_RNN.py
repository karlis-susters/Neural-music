import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from create_featuresets import *
from math import log


now = datetime.now()
logdir = "C:\\MIDI_Gen\\tf_logs\\" + now.strftime("%Y%m%d-%H%M%S") + "\\"
print("logging to", logdir)

hm_epochs = 10
n_classes = 69
#num_batches = 1 #512 
num_batches = 70 

measure_acc_fragments = 10 
save_fragments = 30 
chunk_size = 69
time_steps = 1 #1 for generation, 128/256 for training
layers = 3

rnn_size = 512 
rnn_size2 = 512
rnn_size3 = 512

#height x width#
x = tf.placeholder('float', [None, None, chunk_size], name='x') #input data - 128 timesteps x 69 letters 
y = tf.placeholder('float', [None, chunk_size], name='y')  
init_state = tf.placeholder(tf.float32, [layers, 2, None, rnn_size]) #layers x 2 (c and h) x num_batches x neuron count

def recurrent_neural_network(x):
	layer = {'weights':tf.Variable(tf.random_normal([rnn_size3, n_classes]), trainable=True, name="hl1_w"),
		   'biases':tf.Variable(tf.random_normal([n_classes]), trainable=True, name="hl1_b")} 

	split_by_layers = tf.unstack(init_state, axis=0)
	rnn_tuple_state = tuple(
		[tf.nn.rnn_cell.LSTMStateTuple(split_by_layers[i][0], split_by_layers[i][1]) for i in range(layers)])

	lstm_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_size)
	lstm_cell2 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_size2)
	lstm_cell3 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_size3)

	multi_cell = rnn_cell.MultiRNNCell([lstm_cell, lstm_cell2, lstm_cell3], state_is_tuple=True)
	outputs, states = tf.nn.dynamic_rnn(multi_cell, x, dtype=tf.float32, initial_state = rnn_tuple_state)
	outputs = tf.transpose(outputs, [1, 0, 2])
	output = []
	for i in range(time_steps):
		output.append(tf.add(tf.matmul(outputs[i], layer['weights']), layer['biases']))
	output_tensor = tf.convert_to_tensor(output, dtype = tf.float32)
	output_tensor = tf.transpose(output_tensor, [1, 0, 2])
	return output_tensor, states


def train_neural_network(x):

	train_x, train_y, test_x, test_y = read_pickle()

	batch_size = len(train_x) // num_batches
	time_fragments_in_batch = batch_size // time_steps

	prediction, state = recurrent_neural_network(x)
	prediction = tf.reshape(prediction, [time_steps*num_batches, n_classes])

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) 
	saver = tf.train.Saver()
	print("Running...")

	with tf.Session() as sess:


		sess.run(tf.global_variables_initializer()) 
		test_x = sess.run(tf.reshape(test_x, [num_batches, time_steps]))
		
		try:
			saver.restore(sess, 'C:\\MIDI_Gen)\\')
			print("Loading model...")
		except:
			print("Creating new model...")
			saver.save(sess, 'C:\\MIDI_Gen\\')
		
		summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph) #!!!

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) # Check if prediction equals y
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		acc_summary = tf.summary.scalar("accuracy", accuracy)
		cost_summary = tf.summary.scalar("cost", cost)

		j = 0
		
		test_x_1hot = sess.run(tf.one_hot(test_x, chunk_size))
		test_y_1hot = sess.run(tf.one_hot(test_y, chunk_size))
		for epoch in range(hm_epochs):
			_current_state = np.zeros((layers, 2, num_batches, rnn_size))
			epoch_loss = 0
			i = 0
			print ("Starting epoch {}!".format(epoch))
			for i in range(time_fragments_in_batch):
					print("Time fragment {} / {}".format(i, time_fragments_in_batch))
					x_here = []
					y_here = []
					for k in range(num_batches):
						x_here.append(train_x[i*time_steps + k*batch_size : (i+1)*time_steps + k*batch_size])
						y_here.extend(train_y[i*time_steps + k*batch_size : (i+1)*time_steps + k*batch_size])

					x_here_1hot, y_here_1hot = sess.run([tf.one_hot(x_here, chunk_size), tf.one_hot(y_here, chunk_size)])
					_, c, _current_state, cost_summary_str = sess.run([optimizer, cost, state, cost_summary], feed_dict={x: x_here_1hot, y: y_here_1hot, init_state: _current_state})
					summary_writer.add_summary(cost_summary_str, j)
					epoch_loss += c
					print(epoch_loss, c)

					if i % measure_acc_fragments == 0:
						now = datetime.now()
						print("Time:")
						print(now.strftime("%Y%m%d-%H%M%S"))
						acc_summary_str, acc_str = sess.run([acc_summary, accuracy], 
										  feed_dict={x:test_x_1hot, y:test_y_1hot, init_state:np.zeros((layers, 2, num_batches, rnn_size))})
						print("Accuracy:", acc_str)
						summary_writer.add_summary(acc_summary_str, j)

					if i % save_fragments == 0:
						saver.save(sess, 'C:\\MIDI_Gen\\')
					j += 1
					

			print ('Epoch', epoch, 'completed out of', hm_epochs, ', loss:', epoch_loss)

def adjust_seq(seq):
	if (len(seq) < time_steps):
		seq = [-1 for i in range(time_steps-(len(seq)))] + seq
	else:
		seq = seq[-time_steps:]
	return seq

def generate_text(length, seed, n):
	
	prediction, state_tens = recurrent_neural_network(x)
	saver = tf.train.Saver()
	start = []
	for c in seed:
		if (c == '~'):
			start.append(67)
		elif (c == '\n'):
			start.append(68)
		else:
			start.append(ord(c) - 27)
	print(start)
	res = ""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, 'C:\\MIDI_Gen\\')
		start_x = [sess.run(tf.one_hot(adjust_seq(start[:-1]), n_classes))]
		_, start_state = sess.run([prediction, state_tens], feed_dict={x:np.array(start_x), init_state:np.zeros((layers, 2, 1, rnn_size))})
		sequences = [[start, 1.0, [start_state[0][0][0], start_state[0][1][0], start_state[1][0][0], start_state[1][1][0], start_state[2][0][0], start_state[2][1][0]]]]

		for i in range(length): #for each letter to generate
			all_candidates = []
			test_x = []
			test_states = []
			for j in range(len(sequences)): #expand each sequence
				test_x.append(sequences[j][0][-1])
			
			feed_x = np.zeros((len(test_x), 1, n_classes)) #Get one hot of x data
			feed_x[np.arange(len(test_x)), 0, test_x] = 1

			test_states = [sequences[j][2] for j in range(len(sequences))] #Gets all states from sequqnces
			test_states = np.transpose(test_states, [1, 0, 2]) #Transposes to [layers*2, n, n_classes]
			feed_state = [[test_states[0], test_states[1]], [test_states[2], test_states[3]], [test_states[4], test_states[5]]] #Groups up states by layer

			network_output, state = sess.run([prediction, state_tens], feed_dict={x:feed_x, init_state:feed_state})
			pred_str = tf.transpose(network_output, [1, 0, 2])
			pred_str = pred_str[-1]
			probs = sess.run(tf.nn.softmax(pred_str))
			for j in range(len(sequences)):
				coef = 1/sequences[0][1]
				seq, score, state_here = sequences[j]
				for k in range(n_classes): #consider each next option
					if (probs[j][k] == 1):
						probs[j][k] = 0.9999
					if (score == 0):
						print("what") #Math gone wrong
					candidate = [seq + [k], score * -log(probs[j][k])*coef, 
						  [state[0][0][j], state[0][1][j], state[1][0][j], state[1][1][j], state[2][0][j], state[2][1][j]]]
					if (k == 67):
						candidate[1] = 1e300
					all_candidates.append(candidate)
			print(i)
			ordered = sorted(all_candidates, key=lambda tup:tup[1])
			sequences = ordered[:n]
			print(sequences[0][1])
			print(sequences[0][0][-1])
		print(sequences[0][0])
		return sequences

def create_music(length, outfile):
	seed = ""
	with open("seed.txt", "r") as f:
		for line in f:
			seed = seed + line
	with open(outfile, "w") as f:
		data = generate_text(length, seed, 5)
		str = ""
		for c in data[0][0]:
			if (c == 67):
				str += "~"
			elif (c == 68):
				str += "\n"
			else:
				str += chr(c+27)
		f.write(str)

#read_training_data(n_classes, num_batches*time_steps) #When training, uncomment next two lines
#train_neural_network(x)
create_music(500, "music_NN.csv") #When generating, set time_steps = 1
