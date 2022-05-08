from math import  log, exp, sqrt
from numpy 	import *
import random

def sigmoid(x):
	return 1/(1+ math.exp(-x))

def der_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))


def forward_propagation(activation, weight, z, neuron, bias):

	for i in range(1,len(neuron)):
		for j in range(neuron[i]):
			for k in range(neuron[i-1]):
				z[i][j] += activation[i-1][k] * weight[i][j][k]
			
			z[i][j] += bias[i][j]
			activation[i][j] = sigmoid(z[i][j])

	return activation, z


def back_propagation(weight, error, activation, z, l, neuron, bias):

	for i in range(len(neuron)-1, 0, -1):
		for j in range(neuron[i]):
			for k in range(neuron[i-1]):
				old_weight = weight[i][j][k]
				weight[i][j][k] -= l * -1 *error[i][j] * der_sigmoid(z[i][j]) * activation[i-1][k]
				error[i-1][k] += error[i][j] * old_weight
	
			bias[i][j] -= l * -1 *(error[i][j]	* der_sigmoid(z[i][j]))

	return weight, bias


def loss_calculation(output, predicted, error, n):
	loss = 0
	for i in range(len(predicted)):
		error[n-1][i] = (output - predicted[i])
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
			for k in range(neuron[i-1]):
				activation_j.append(random.randrange(-2, 2))
			layer.append(activation_j)
			bias_weight.append(random.randrange(-2, 2))
		weight.append(layer)
		bias.append(bias_weight)

	print('Activation: ', activation)

	print('Weight: ', weight)

	print('Bias: ', bias)

	#data = genfromtxt("or_gate.csv",delimiter=",")
	#data = genfromtxt("and_gate.csv",delimiter=",")
	#data = genfromtxt("not_gate.csv",delimiter=",")
	data = genfromtxt("xor_gate.csv",delimiter=",")
	#data = genfromtxt("nand_gate.csv",delimiter=",")
	#data = genfromtxt("nor_gate.csv",delimiter=",")
	#data=genfromtxt("iris_dataset.csv",delimiter=",")

	# for normal gates, learning_rate 0.1 and 10000
	learning_rate = 0.01

	# for exor gate
	#learning_rate = 0.0007

	for epoch in range(10000):
		for i in range(len(data)):
			for l in range(n):
				for m in range(neuron[l]):
					error[l][m] = 0
					z[l][m] = 0

			activation[0] = data[i, 0:-1]
			output = data[i,-1]
			activation, z = forward_propagation(activation, weight, z, neuron, bias)

			#calculate loss and error
			loss, error = loss_calculation(output, activation[n-1], error, len(neuron))

			########################################
			# normal gradient
			########################################
			weight, bias = back_propagation(weight, error, activation, z, learning_rate, neuron, bias)

	print('**************************')
	print('Bias: ', bias)
	print('Weight: ', weight)
	
	cost = 0
	for i in range(len(data)):
		print(i,end=' ')
		for l in range(n):
			for m in range(neuron[l]):
				z[l][m] = 0

		activation[0] = data[i, 0:-1]
		output = data[i,-1]
		print(activation[0],end='  ')

		activation, z = forward_propagation(activation, weight, z, neuron, bias)
		print(activation[n-1][0])

		cost += (output - round(activation[n-1][0]))**2

	print('Cost:', cost/len(data))

if __name__ == '__main__':
	neural_network()

