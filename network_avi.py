
import random

import numpy as np

class Network_new(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes);
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[-1:], sizes[1:])]

    def feedForwardAvi(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w*a+b))
        return a;

    def SGD(self, training_data, epochs, mini_batch_size, eta, testdata=None):
        if(testdata): n_test = len(testdata)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k,k+mini_batch_size] for k in xrange( 0, n, mini_batch_size)]
            for min_batch in mini_batches:
                self.update_mini_batch(min_batch, eta)

    def update_mini_batch(self, min_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in min_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(min_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(min_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self , x, y):
        activation = x;
        activations = [x]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(z)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(zs(-l))
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w




    def cost_derivative(self, x, y):
        return (x - y)










def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))





