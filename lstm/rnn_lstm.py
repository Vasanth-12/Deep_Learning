# LSTM cell

import random
from math import  log, exp
from numpy 	import *
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import expit, logit

def sigmoid(x):
	#if x <= -700:
	#	return 0 
	return 1/(1 + exp(-x))


def tanh(x):
	#if x <= -700:
	#	return -1
	#if x >= 700:
	#	return 1
	return (math.exp(x) - math.exp(-(x))) / (math.exp(x) + math.exp(-(x)))
	#return 2/(1+math.exp(-(2*x))) - 1


def forward_propagation(hidden_layer,
			forgot_gate, forgot_gate_z, forgot_w, forgot_u, forgot_bias_w,
			input_gate, input_gate_z, input_w, input_u, input_bias_w,
			output_gate, output_gate_z, output_w, output_u, output_bias_w,
			candidate, candidate_z, candidate_w, candidate_u, candidate_bias_w,
			cellstate,
			t, input_data):
	length_input = len(input_data)
	length_hidden = len(hidden_layer[0])

	start = time.time()

	forgot_gate_z[t] = forgot_w.dot(input_data) + forgot_u.dot(hidden_layer[t-1]) + forgot_bias_w
	input_gate_z[t] = input_w.dot(input_data) + input_u.dot(hidden_layer[t-1]) + input_bias_w
	output_gate_z[t] = output_w.dot(input_data) + output_u.dot(hidden_layer[t-1]) + output_bias_w
	candidate_z[t] = candidate_w.dot(input_data) + candidate_u.dot(hidden_layer[t-1]) + candidate_bias_w

	forgot_gate[t] = expit(forgot_gate_z[t])
	input_gate[t] = expit(input_gate_z[t])
	output_gate[t] = expit(output_gate_z[t])
	candidate[t] = np.tanh(candidate_z[t])

	cellstate[t] = (cellstate[t] * forgot_gate[t]) + (input_gate[t] * candidate[t])
	hidden_layer[t] = output_gate[t] * np.tanh(cellstate[t])

	end = time.time()


def predict_y(hidden_layer, output_layer, v, bias_v):

	length_hidden = len(v[0])
	length_output = len(v)

	y = np.zeros(length_output)
	z_y = np.zeros(length_output)
	for i in range(length_output):
		z = 0
		for j in range(length_hidden):
			z_y[i] += hidden_layer[j] * v[i][j]

		#z += bias_v[i]
		y[i] = tanh(z_y[i])
		#z_y.append(z)

	return y, z_y


def backPropagation(hidden_layer, error_hidden_layer, 
		output_gate, error_output_gate, output_gate_z, output_w, output_u, output_bias_w,
		cellstate, error_cellstate, 
		candidate, error_candidate, candidate_z, candidate_w, candidate_u, candidate_bias_w,
		input_gate, error_input_gate, input_gate_z, input_w, input_u, input_bias_w,
		forgot_gate, error_forgot_gate, forgot_gate_z, forgot_w, forgot_u, forgot_bias_w,
		t, input_data, l):

	length_input = len(input_data)
	length_hidden = len(hidden_layer[0])

	previous_forgot_u = copy(forgot_u)
	previous_input_u = copy(input_u)
	previous_output_u = copy(output_u)
	previous_candidate_u = copy(candidate_u)

	error_cellstate[t] = error_hidden_layer[t] * output_gate[t] * (1 - np.tanh(cellstate[t])**2)
	error_output_gate[t] = error_hidden_layer[t] * np.tanh(cellstate[t])
	error_candidate[t] = error_cellstate[t] * input_gate[t]
	error_input_gate[t] = error_cellstate[t] * candidate[t]
	error_forgot_gate[t] = error_cellstate[t] * cellstate[t-1]
	error_cellstate[t-1] = error_cellstate[t] * forgot_gate[t]

	forgot_w = l * (error_forgot_gate[t] * (1 - expit(forgot_gate_z[t])) * expit(forgot_gate_z[t])).dot(input_data.transpose())
	forgot_u = l * (error_forgot_gate[t] * (1 - sigmoid(forgot_gate_z[t])) * sigmoid(forgot_gate_z[t])).dot(hidden_layer[t-1].transpose())
	error_hidden_layer[t-1] = previous_forgot_u.dot(error_forgot_gate[t] * (1 - expit(forgot_gate_z[t])) * expit(forgot_gate_z[t]))

	input_w = l * (error_input_gate[t] * (1 - expit(input_gate_z[t])) * expit(input_gate_z[t])).dot(input_data.transpose())
	input_u = l * (error_input_gate[t] * (1 - expit(input_gate_z[t])) * expit(input_gate_z[t])).dot(hidden_layer[t-1].transpose())
	error_hidden_layer[t-1] = previous_input_u.dot(error_input_gate[t] * (1 - expit(input_gate_z[t])) * expit(input_gate_z[t]))

	output_w = l * (error_output_gate[t] * (1 - expit(output_gate_z[t])) * expit(output_gate_z[t])).dot(input_data.transpose())
	output_u = l * (error_output_gate[t] * (1 - expit(output_gate_z[t])) * expit(output_gate_z[t])).dot(hidden_layer[t-1].transpose())
	error_hidden_layer[t-1] = previous_output_u.dot(error_output_gate[t] * (1 - expit(output_gate_z[t])) * expit(output_gate_z[t]))

	candidate_w = l * (error_candidate[t] * (1 - np.tanh(candidate_z[t])**2)).dot(input_data.transpose())
	candidate_u = l * (error_candidate[t] * (1 - np.tanh(candidate_z[t])**2)).dot(hidden_layer[t-1].transpose())
	error_hidden_layer[t-1] = previous_candidate_u.dot(error_candidate[t] * (1 - np.tanh(candidate_z[t])**2))

	forgot_bias_w = l * error_forgot_gate[t] * (1 - expit(forgot_gate_z[t])) * expit(forgot_gate_z[t])
	input_bias_w = l * error_input_gate[t] * (1 - expit(input_gate_z[t])) * expit(input_gate_z[t])
	output_bias_w = l * error_output_gate[t] * (1 - expit(output_gate_z[t])) * expit(output_gate_z[t])
	candidate_bias_w = l * error_candidate[t] * (1 - np.tanh(candidate_z[t])**2)

	
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
		X.append(sin_wave[i:i+seq_len])
		Y.append(sin_wave[i+seq_len])
	    
	X = np.array(X)
	X = np.expand_dims(X, axis=2)

	Y = np.array(Y)
	Y = np.expand_dims(Y, axis=1)

	# one hidden layer => no. of neurons in the hidden layer
	hidden = 15
	hidden_layer = np.zeros((sequence_input, hidden, 1))
	error_hidden_layer = np.zeros((sequence_input, hidden, 1))
	z = np.zeros((sequence_input, hidden, 1))
	cellstate = np.zeros((sequence_input, hidden, 1))
	forgot_gate = np.zeros((sequence_input, hidden, 1))
	input_gate = np.zeros((sequence_input, hidden, 1))
	output_gate = np.zeros((sequence_input, hidden, 1))
	candidate = np.zeros((sequence_input, hidden, 1))
	forgot_gate_z = np.zeros((sequence_input, hidden, 1))
	cellstate_z = np.zeros((sequence_input, hidden, 1))
	input_gate_z = np.zeros((sequence_input, hidden, 1))
	output_gate_z = np.zeros((sequence_input, hidden, 1))
	candidate_z = np.zeros((sequence_input, hidden, 1))
	error_cellstate = np.zeros((sequence_input, hidden, 1))
	error_forgot_gate = np.zeros((sequence_input, hidden, 1))
	error_input_gate = np.zeros((sequence_input, hidden, 1))
	error_output_gate = np.zeros((sequence_input, hidden, 1))
	error_candidate = np.zeros((sequence_input, hidden, 1))
	

	# output neurons
	output_layer = 1
	output_neurons = np.zeros((1))
	error_output_neurons = np.zeros((1))

	# w -> weight between input and hidden
	forgot_w = np.random.uniform(-0.3, 0.3, (hidden, sequence_input))
	forgot_bias_w = np.random.uniform(-0.1, 0.1, (hidden, 1))

	input_w = np.random.uniform(-0.3, 0.3, (hidden, sequence_input))
	input_bias_w = np.random.uniform(-0.1, 0.1, (hidden, 1))

	output_w = np.random.uniform(-0.3, 0.3, (hidden, sequence_input))
	output_bias_w = np.random.uniform(-0.1, 0.1, (hidden, 1))

	candidate_w = np.random.uniform(-0.3, 0.3, (hidden, sequence_input))
	candidate_bias_w = np.random.uniform(-0.1, 0.1, (hidden, 1))

	# v -> weight between hidden to output
	v = np.random.uniform(-0.3, 0.3, (output_layer, hidden))
	bias_v = np.random.uniform(-0.1, 0.1, (output_layer, 1))

	# u -> weight between hidden to hidden
	forgot_u = np.random.uniform(-0.3, 0.3, (hidden, hidden))

	input_u = np.random.uniform(-0.3, 0.3, (hidden, hidden))

	output_u = np.random.uniform(-0.3, 0.3, (hidden, hidden))

	candidate_u = np.random.uniform(-0.3, 0.3, (hidden, hidden))
	
	epoch = 100
	l = 0.7
	for i in range(epoch):
		print("--------------------",i,"--------------------")
		for j in range(len(X)):
		#for j in range(1):
			#print(j)
			input_len = len(X[j])
			
			hidden_layer = np.zeros((sequence_input, hidden, 1))
			error_hidden_layer = np.zeros((sequence_input, hidden, 1))
			z = np.zeros((sequence_input, hidden, 1))
			cellstate = np.zeros((sequence_input, hidden, 1))
			forgot_gate = np.zeros((sequence_input, hidden, 1))
			input_gate = np.zeros((sequence_input, hidden, 1))
			output_gate = np.zeros((sequence_input, hidden, 1))
			candidate = np.zeros((sequence_input, hidden, 1))
			forgot_gate_z = np.zeros((sequence_input, hidden, 1))
			cellstate_z = np.zeros((sequence_input, hidden, 1))
			input_gate_z = np.zeros((sequence_input, hidden, 1))
			output_gate_z = np.zeros((sequence_input, hidden, 1))
			candidate_z = np.zeros((sequence_input, hidden, 1))
			error_cellstate = np.zeros((sequence_input, hidden, 1))
			error_forgot_gate = np.zeros((sequence_input, hidden, 1))
			error_input_gate = np.zeros((sequence_input, hidden, 1))
			error_output_gate = np.zeros((sequence_input, hidden, 1))
			error_candidate = np.zeros((sequence_input, hidden, 1))

			#for t in range(0, sequence_input):
			t = 0
			while  t<input_len:

				input_data = np.zeros((sequence_input, 1))
				#
				input_data[t][0] = X[j][t]

				forward_propagation(hidden_layer,
						forgot_gate, forgot_gate_z, forgot_w, forgot_u, forgot_bias_w,
						input_gate, input_gate_z, input_w, input_u, input_bias_w,
						output_gate, output_gate_z, output_w, output_u, output_bias_w,
						candidate, candidate_z, candidate_w, candidate_u, candidate_bias_w,
						cellstate, t, input_data)
				t += 1
			
			# prediction
			y, z_y = predict_y(hidden_layer[sequence_input-1], output_neurons, v, bias_v)

			#print("Epoch: ", i,"Y: ",Y[j],"\ty^: ",y)
			# Loss calculation
			loss = (Y[j] - y[0])**2 / 2
			#print("Loss: ",loss)
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
			#for t in range(sequence_input-1 , -1, -1):
			t = input_len - 1
			while t > -1: 
				input_data = np.zeros((sequence_input, 1))
				#
				input_data[t][0] = X[j][t]

				backPropagation(hidden_layer, error_hidden_layer, 
						output_gate, error_output_gate, output_gate_z, output_w, output_u, output_bias_w,
						cellstate, error_cellstate, 
						candidate, error_candidate, candidate_z, candidate_w, candidate_u, candidate_bias_w,
						input_gate, error_input_gate, input_gate_z, input_w, input_u, input_bias_w,
						forgot_gate, error_forgot_gate, forgot_gate_z, forgot_w, forgot_u, forgot_bias_w,
						t, input_data, l)

				t -= 1

	print("Success")
	# test the model
	
	loss = 0
	_Y = []
	for j in range(len(X)):
			#print(j)
		input_len = len(X[j])
		
		hidden_layer = np.zeros((sequence_input, hidden, 1))
		error_hidden_layer = np.zeros((sequence_input, hidden, 1))
		z = np.zeros((sequence_input, hidden, 1))
		cellstate = np.zeros((sequence_input, hidden, 1))
		forgot_gate = np.zeros((sequence_input, hidden, 1))
		input_gate = np.zeros((sequence_input, hidden, 1))
		output_gate = np.zeros((sequence_input, hidden, 1))
		candidate = np.zeros((sequence_input, hidden, 1))
		forgot_gate_z = np.zeros((sequence_input, hidden, 1))
		cellstate_z = np.zeros((sequence_input, hidden, 1))
		input_gate_z = np.zeros((sequence_input, hidden, 1))
		output_gate_z = np.zeros((sequence_input, hidden, 1))
		candidate_z = np.zeros((sequence_input, hidden, 1))
		error_cellstate = np.zeros((sequence_input, hidden, 1))
		error_forgot_gate = np.zeros((sequence_input, hidden, 1))
		error_input_gate = np.zeros((sequence_input, hidden, 1))
		error_output_gate = np.zeros((sequence_input, hidden, 1))
		error_candidate = np.zeros((sequence_input, hidden, 1))

		#for t in range(0, sequence_input):
		t = 0
		while  t<input_len:

			input_data = np.zeros((sequence_input, 1))
			#
			input_data[t][0] = X[j][t]

			forward_propagation(hidden_layer,
					forgot_gate, forgot_gate_z, forgot_w, forgot_u, forgot_bias_w,
					input_gate, input_gate_z, input_w, input_u, input_bias_w,
					output_gate, output_gate_z, output_w, output_u, output_bias_w,
					candidate, candidate_z, candidate_w, candidate_u, candidate_bias_w,
					cellstate, t, input_data)
			t += 1
		
		# prediction
		y, z_y = predict_y(hidden_layer[sequence_input-1], output_neurons, v, bias_v)

		print("Epoch: ", i,"Y: ",Y[j],"\ty^: ",y)
		_Y.append(y[0])

		# Loss calculation
		loss = (Y[j] - y[0])**2 / 2
		print("Loss: ",loss)

	plt.plot(_Y, 'b')
	plt.plot(Y, 'r')
	plt.show()
	

if __name__ == '__main__':
	rNN()