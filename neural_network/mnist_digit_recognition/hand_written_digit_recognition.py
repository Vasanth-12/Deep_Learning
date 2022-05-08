#neural network for handwritten digit recognition

from math import  log, exp, sqrt
from numpy 	import *
import random
import pandas as pd

def sigmoid(x):
	return 1/(1+ math.exp(-x))

def der_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))


def forward_propagation(activation, weight, z, neuron, bias):

	for i in range(1,len(neuron)):
		for j in range(neuron[i]):
			for k in range(neuron[i-1]):
				#print('I ',i,' J: ',j,' K: ',k)
				#print(' WEIGHT:',weight[i][j][k])
				#print('ACTIVATION: ',activation[i-1][k])
				z[i][j] += activation[i-1][k] * weight[i][j][k]
			
			z[i][j] += bias[i][j]
			#print(z[i][j])
			#activation[i][j] = sigmoid(z[i][j])
			try:
				activation[i][j] = sigmoid(z[i][j])
			except OverflowError:
				activation[i][j] = 0

	return activation, z


def back_propagation(weight, error, activation, z, l, neuron, bias):

	for i in range(len(neuron)-1, 0, -1):
		for j in range(neuron[i]):
			for k in range(neuron[i-1]):
				old_weight = weight[i][j][k]
				#print('Z: ',z[i][j])
				#weight[i][j][k] -= l * -1 *error[i][j] * der_sigmoid(z[i][j]) * activation[i-1][k]
				try:
					weight[i][j][k] -= l * -1 *error[i][j] * der_sigmoid(z[i][j]) * activation[i-1][k]
				except OverflowError:
					weight[i][j][k] = weight[i][j][k]

				error[i-1][k] += error[i][j] * old_weight
	
			#bias[i][j] -= l * -1 *(error[i][j]	* der_sigmoid(z[i][j]))
			try:
				bias[i][j] -= l * -1 *(error[i][j]	* der_sigmoid(z[i][j]))
			except OverflowError:
				bias[i][j] = bias[i][j]

	return weight, bias


def loss_calculation(output, predicted, error, n):
	loss = 0
	actual = []

	#one hot encoded list
	for i in range(len(predicted)):
		if i == output:
			actual.append(1)
		else:
			actual.append(0)
		
	#print("ACTUAL: ", actual)
	#print("PREDICTED: ", predicted)

	for i in range(len(predicted)):
		error[n-1][i] = (actual[i] - predicted[i])
		loss += error[n-1][i]**2 

	return loss/2, error


def neural_network():
	neuron=[]
	activation=[]
	error=[]
	weight=[]
	z=[]
	bias = []

	n = int(input('No of layers:\n'))

	for i in range(n):
		print('No. of neuron in ',i,' layer:\t')
		neuron.append(int(input()))

	for l in range(n):
		activation_neuron = []
		error_neuron = []
		z_neuron = []
		for m in range(neuron[l]):
			activation_neuron.append(0)
			error_neuron.append(0)
			z_neuron.append(0)
		activation.append(activation_neuron)
		error.append(error_neuron)
		z.append(z_neuron)

	weight.append(0)
	bias.append(0)
	for i in range(1,n):
		layer = []
		bias_weight = []
		for j in range(neuron[i]):
			activation_j = []
			previous_layer = neuron[i-1]
			weight_init = 1/ (previous_layer) ** (1/2)
			for k in range(previous_layer):
				activation_j.append(random.uniform(-weight_init, weight_init))
			layer.append(activation_j)
			bias_weight.append(random.randrange(-1, 1))
		weight.append(layer)
		bias.append(bias_weight)

	#print('Activation: ', activation)

	#print('Weight: ', weight)

	#print('Bias: ', bias)

	data=pd.read_csv("mnist_train.csv",sep=",", header=None).dropna().sort_values(by=[784])
	data /= 255

	# for normal gates, learning_rate 0.1 and 10000
	learning_rate = 0.1

	# for exor gate
	#learning_rate = 0.0007

	for epoch in range(5):
		for i in range(6000):
			count = i*10
			random_pick = random.randrange(count, count+10)
			print(random_pick)
			for l in range(n):
				for m in range(neuron[l]):
					error[l][m] = 0
					z[l][m] = 0

			activation[0] = data.loc[random_pick, 0:783]
			output = data.loc[random_pick,784] * 255
			activation, z = forward_propagation(activation, weight, z, neuron, bias)

			#calculate loss and error
			loss, error = loss_calculation(output, activation[n-1], error, len(neuron))

			########################################
			# normal gradient
			########################################
			weight, bias = back_propagation(weight, error, activation, z, learning_rate, neuron, bias)


	data=pd.read_csv("mnist_test.csv",sep=",", header=None).dropna().sort_values(by=[784])
	data /= 255
	cost = 0
	for i in range(1000):
		count = i*10
		random_pick = random.randrange(count, count+10)
		print(random_pick,end=' ')
		for l in range(n):
			for m in range(neuron[l]):
				z[l][m] = 0

		activation[0] = data.loc[i, 0:783]
		output = data.loc[i,784] * 255
		#print(activation[0],end='  ')

		activation, z = forward_propagation(activation, weight, z, neuron, bias)

		actual = []
		for ab in range(len(activation[n-1])):
			if ab == output:
				actual.append(1)
			else:
				actual.append(0)

		max = 0
		for ab in range(len(activation[n-1])):
			if activation[n-1][ab] > max:
				max = activation[n-1][ab]

		pivot = -1
		for ab in range(len(activation[n-1])):
			if activation[n-1][ab] == max:
				activation[n-1][ab] = 1
				pivot = ab
			else:
				activation[n-1][ab] = 0


		print(pivot,' -> ',end=' ')
		print(output)
		print()

		for ab  in range(len(activation[n-1])):
			cost += (actual[ab] - round(activation[n-1][ab]))**2


	print('**************************')
	print('Cost:', cost/1000)
	#print('Bias: ', bias)
	#print('Weight: ', weight)

if __name__ == '__main__':
	neural_network()

