import numpy as np
#import matplotlib.pyplot as plt

#neurons: vector fila de n dimensiones con las neuronas en cada capa, donde n es el número de capas.
#neurons = np.array([5,1])


def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

def forward(x, w1, w2, predict = False):
	bias = np.ones((np.shape(x)[0], 1))		# np.shape(X)[0] nos da el número de observaciones
	#x = np.concatenate((bias, x), axis = 1)
z1 = np.dot(x,w1)
a1 = sigmoid(z1)
bias = np.ones((np.shape(a1)[0], 1))
#a1 = np.concatenate((bias, a1), axis = 1)
z2 = np.dot(a1,w2)
a2 = sigmoid(z2)
	return a2

def cost(a2):
	m = np.shape(X)[0]
	loss = (a2 - y)**2
	#loss = -y*np.log(a2) - (1 - y) * np.log(1 - a2)
	cost = np.sum(loss)/m
	return(cost)

def backprop(m):
	db2 = 2*(a2-y[m]) * sigmoid_deriv(z2)
	dw2 = db2 * a2
	db1 = (db2 * w2.T * sigmoid_deriv(z1)).T
	dw1 = np.dot(db1, X)

#X[0][np.newaxis]
	return(dw1,dw2,db1,db2)

#====================================================================================================

X = np.array([[1, 0],
			[0, 1],
			[0, 0],
			[1, 1]])

y = np.array([[1], [1], [0], [0]])

#====================================================================================================

w1 = np.random.randn(np.shape(X)[1] , 5) #np.shape(X)[1] nos da el número de variables en cada observación (+1 por bias?), 5 vendrá determinado por neurons
w2 = np.random.randn(5, 1) #6 es el 5 de neurons + la columna de bias

#====================================================================================================

predicts = forward(X,w1,w2)
print(predicts)

print(cost(predicts))


