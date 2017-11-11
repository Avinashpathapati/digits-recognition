import random
import numpy as np

class CostEntropyCost(object):
   @staticmethod
   def fn(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)));
   @staticmethod
   def delta(z, a, y):
       return a-y;

class Network_new(object):

    def default_weight_initializer(self):
        self.weights = [np.random.rand(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])];
        self.biases = [np.random.rand(y,1) for y in self.sizes[1:]];

    def __init__(self, sizes, cost=CostEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def feedword(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)

        return a;

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
            evaluation_data = None, monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False,
            monitor_training_accuracy = False,
            monitor_training_cost = False
            ):
        if(evaluation_data):
            n_data = len(evaluation_data)

        n = len(training_data)

        for j in xrange(epochs) :
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)


    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape)for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,y);
            nabla_b = [nb+dnb for nb, dnb in zip(self.nabla_b, delta_nabla_b)];
            nabla_w = [nw + dnw for nw, dnw in zip(self.nabla_w, delta_nabla_w)];

        self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w , nw in zip(self.weights, nabla_w)]
        self.biases = [ b - (eta / len(mini_batch)) * nb for b, nb in
                        zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost.delta(z, activation, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = activations[-l]
            sp = sigmoid_prime(z)
            delta = (np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(sp))
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)
    def total_cost(self, data, lmbda, convert = False):
        cost = 0.0
        for x,y in data:
            a = self.feedword(x);
            if convert: y = vectorized_result(y)
            cost +=  self.cost.fn(a, y)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost


def vectorized_result(j):
    vector_a = np.zeros(10,1)
    vector_a[j] = 1;
    return vector_a
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


























