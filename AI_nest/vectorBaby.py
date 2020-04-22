import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return sigmoid(x) * (1 - sigmoid(x))

X = np.array([[1, 0],
			[0, 1],
			[0, 0],
			[1, 1]])

y = np.array([[1], [1], [0], [0]])


X = np.rot90(X[np.newaxis])
y = np.rot90(y[np.newaxis])


#start

w1 = np.random.randn(np.shape(X)[2] , 5)
w2 = np.random.randn(5, 1)
b1 = np.random.randn(1,1,5)
b2 = np.random.randn(1,1,1)

w1 = w1[np.newaxis]
w2 = w2[np.newaxis]

m = np.shape(X)[0]
lr = 0.05
epochs = 15000
clist = []

for x in range(epochs):
	#forwardprop
	#n = neurons per layer: n[1] will give the number of neurons in the first layer
	z1 = np.matmul(X, w1) + b1 #(m, n[1], n[2])
	a1 = sigmoid(z1)
	z2 = np.matmul(a1, w2) + b2
	a2 = sigmoid(z2)
	#cuidao np.rot o np.transpose??
	#backprop
	dloss = np.array([[[2]]]) * (a2-y)
	db2 = dloss * sigmoid_deriv(z2) # = np.matmul(dloss,sigmoid_deriv(z2))
	dw2 = np.matmul(np.transpose(a1, (0,2,1)), db2)
	db1 = db2 * sigmoid_deriv(z1) * np.transpose(w2, (0,2,1))
	dw1 = np.matmul(np.transpose(X, (0,2,1)), db1)
	#FLATTEN
	db2 = db2.sum(axis=0)
	dw2 = dw2.sum(axis=0)
	db1 = db1.sum(axis=0)
	dw1 = dw1.sum(axis=0)
	#descent
	b2=b2-lr*db2
	w2=w2-lr*dw2
	b1=b1-lr*db1
	w1=w1-lr*dw1
	#cost
	loss = (a2 - y)**2
	#	loss = -y*np.log(a2) - (1 - y) * np.log(1 - a2)
	cost = np.sum(loss)/m
	clist.append(cost)
	print(str(epochs-x) + ": cost = " + str(cost))

print(a2)
plt.plot(clist)
plt.show()




'''

a1 4x5x1
z2 4x1x1




>>> np.rot90(w1, k=0)[0,1,0]
1.1997270386584933
>>> np.rot90(w1, k=1)[0,0,0]
1.1997270386584933
>>> np.rot90(w1, k=2)[0,0,0]
1.1997270386584933
>>> np.rot90(w1, k=3)[1,0,0]
1.1997270386584933
>>> np.rot90(w1, k=4)[0,1,0]
1.1997270386584933
>>> w1.shape
(1, 2, 5)
'''



