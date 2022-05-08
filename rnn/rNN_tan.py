#neural_network

import random
from math import  log, exp
from numpy 	import *
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	if x <= -700:
		return -1
	if x >= 700:
		return 1
	return (math.exp(x) - math.exp(-(x))) / (math.exp(x) + math.exp(-(x)))
	#return 2/(1+math.exp(-(2*x))) - 1

def forward_propagation(input_data, hidden_layer, z, w, u, t, bias_w, bias_u):

	length_input = len(w[0])
	length_hidden = len(u)

	for i in range(length_hidden):
		# input to hidden
		for j in range(length_input):
			z[t][i] += w[i][j] * input_data[j]

		# previous hidden to hidden
		for j in range(length_hidden):
			z[t][i] += u[i][j] * hidden_layer[t-1][j]

		z[t][i] += bias_w[i] + bias_u[i]

		hidden_layer[t][i] = sigmoid(z[t][i])

	return hidden_layer, z


def predict_y(hidden_layer, output_layer, v, bias_v):

	length_hidden = len(v[0])
	length_output = len(v)

	y = []
	z_y = []
	for i in range(length_output):
		z = 0
		for j in range(length_hidden):
			z += hidden_layer[j] * v[i][j]

		z += bias_v[i]
		y.append(sigmoid(z))
		z_y.append(z)

	return y, z_y


def backward_propagation(w, u, l, t, hidden_layer, z, input_data, error_hidden_layer, bias_w, bias_u):

	previous_w = w
	previous_u = u

	length_hidden = len(u)
	length_input = len(w[0])

	for i in range(length_hidden):
		abc = error_hidden_layer[t][i] * (1-sigmoid(z[t][i])**2)#((1 - sigmoid(z[t][i])) * sigmoid(z[t][i]))
		for j in range(length_input):
			w[i][j] += l * abc * input_data[j]

		for j in range(length_hidden):
			u[i][j] += l * abc * hidden_layer[t-1][j]
			error_hidden_layer[t-1][j] += abc * previous_u[i][j]

		bias_w[i] += l * abc
		bias_u[i] += l * abc

	return w, u, error_hidden_layer, bias_w, bias_u


def rNN():
	# sequence input
	sequence_input = 50

	sin_wave = np.array([math.sin(x) for x in np.arange(200)])

	seq_len = 50
	num_records = len(sin_wave) - seq_len
	#print(seq_len,"   ",num_records)

	X = []
	Y = []

	for i in range(num_records - 50):
		a = sin_wave[i:i+seq_len].tolist()
		b = sin_wave[i+seq_len].tolist()
		X.append(a)
		Y.append(b)

	# one hidden layer => no. of neurons in the hidden layer
	hidden = 20
	hidden_layer = []
	error_hidden_layer = []
	z = []
	for t in range(sequence_input):
		temp_hidden = []
		temp_error = []
		temp_z = []
		for i in range(hidden):
			temp_hidden.append(0)
			temp_error.append(0)
			temp_z.append(0)
		hidden_layer.append(temp_hidden)
		error_hidden_layer.append(temp_error)
		z.append(temp_z)

	# output neurons
	output_layer = 1
	output_neurons = []
	error_output_neurons = []
	for i in range(output_layer):
		output_neurons.append(0)
		error_output_neurons.append(0)

	# w -> weight between input and hidden
	w = []
	bias_w = []
	for i in range(hidden):
		temp = []
		for j in range(sequence_input):
			temp.append(random.uniform(-0.1,0.1))
		w.append(temp)
		bias_w.append(random.uniform(-0.1,0.1))

	# v -> weight between hidden to output
	v = []
	bias_v = []
	for i in range(output_layer):
		temp = []
		for j in range(hidden):
			temp.append(random.uniform(-0.1,0.1))
		v.append(temp)
		bias_v.append(random.uniform(-0.1,0.1))

	# u -> weight between hidden to hidden
	u = []
	bias_u = []
	for i in range(hidden):
		temp = []
		for j in range(hidden):
			temp.append(random.uniform(-0.1,0.1))
		u.append(temp)
		bias_u.append(random.uniform(-0.1,0.1))

	epoch = 100
	l = 0.0007
	for i in range(epoch):
		print(i)
		for j in range(len(X)):
		#for j in range(2):
			for t in range(sequence_input):
				for xyz in range(hidden):
					z[t][xyz] = 0
					hidden_layer[t][xyz] = 0
					error_hidden_layer[t][xyz] = 0
			for t in range(0, sequence_input):

				input_data = [0] * sequence_input
				input_data[t] = X[j][t]

				# rNN forward propagation and predict output
				hidden_layer, z = forward_propagation(input_data, hidden_layer, z, w, u, t, bias_w, bias_u)

			# prediction
			y, z_y = predict_y(hidden_layer[sequence_input-1], output_neurons, v, bias_v)

			##print("Epoch: ", i,"Y: ",Y[j],"\ty^: ",y)
			# Loss calculation
			loss = (Y[j] - y[0])**2 / 2
			##print("Loss: ",loss)
			for k in range(output_layer):
				error_output_neurons[k] = Y[j] - y[k]

			previous_v = v
			for k in range(output_layer):
				abc = error_output_neurons[k] * (1 - sigmoid(z_y[k])**2)#((1 - sigmoid(z_y[k])) * sigmoid(z_y[k]))
				for lm in range(hidden):
					error_hidden_layer[sequence_input-1][lm] += abc * previous_v[k][lm]
					v[k][lm] += l * abc * hidden_layer[sequence_input-1][lm]

				bias_v[k] += l * abc

			#rNN backward propagation
			for t in range(sequence_input-1 , -1, -1):
				input_data = [0] * sequence_input
				input_data[t] = X[j][t]

				w, u, error_hidden_layer, bias_w, bias_u = backward_propagation(w, u, l, t, hidden_layer, z, input_data, error_hidden_layer, bias_w, bias_u)

	print("*********************************************")
	print("W: ", w)
	print("*********************************************")
	print()

	print("*********************************************")
	print("U: ", u)
	print("*********************************************")
	print()

	print("*********************************************")
	print("V: ", v)
	print("*********************************************")
	print()

	print("Success")
	# test the model
	loss = 0
	_Y = []
	for j in range(len(X)):
		for t in range(sequence_input):
			for xyz in range(hidden):
				z[t][xyz] = 0
				hidden_layer[t][xyz] = 0
				error_hidden_layer[t][xyz] = 0
		for t in range(0, sequence_input):
			input_data = [0] * sequence_input
			input_data[t] = X[j][t]

			# rNN forward propagation and predict output
			hidden_layer, z = forward_propagation(input_data, hidden_layer, z, w, u, t, bias_w, bias_u)

		# prediction
		y, z_y = predict_y(hidden_layer[sequence_input-1], output_neurons, v, bias_v)

		print("Y: ",Y[j],"\ty^: ",y)
		# Loss calculation
		_Y.append(y[0])
		loss += (Y[j] - y[0])**2 / 2
	print("Total Loss: ",loss/len(X))

	plt.plot(_Y, 'g')
	plt.plot(Y, 'r')
	plt.show()


if __name__ == '__main__':
	rNN()