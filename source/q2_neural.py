import numpy as np
import random
import unittest
import time
from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    # ## Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # ## YOUR CODE HERE: forward propagation
    N = data.shape[0]
    all_mu = data.dot(W1) + b1
    all_h = sigmoid(all_mu)
    all_theta = all_h.dot(W2) + b2
    all_y_hat = softmax(all_theta)
    all_costs = np.sum(labels * np.log(all_y_hat), 1) * -1
    cost = np.mean(all_costs)
    # ## END YOUR CODE

    # ## YOUR CODE HERE: backward propagation
    subtraction = all_y_hat - labels
    E = np.dot(W2, subtraction.T)
    sig_mu = sigmoid_grad(sigmoid(all_mu.T))
    E_sig_mu_mult = E * sig_mu
    gradW1 = np.dot(data.T, E_sig_mu_mult.T) * 1/N
    gradb1 = np.sum(E_sig_mu_mult, 1) * 1/N
    gradW2 = np.dot(all_h.T, subtraction) * 1/N
    gradb2 = np.sum(subtraction.T, 1) * 1/N
    # ## END YOUR CODE

    # ## Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(),
                           gradb1.flatten(),
                           gradW2.flatten(),
                           gradb2.flatten()))
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
                                                         dimensions), params)


class TestNN(unittest.TestCase):

    def test_data_sizes(self):
        random_data_sizes = np.random.randint(1, 100, 5)
        for i in range(5):
            start = time.time()
            N = random_data_sizes[i]
            print("data size = {}".format(N))
            dimensions = [10, 5, 10]
            data = np.random.randn(N, dimensions[0])
            labels = np.zeros((N, dimensions[2]))
            for i in range(N):
                labels[i, random.randint(0, dimensions[2]-1)] = 1
            params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
                dimensions[1] + 1) * dimensions[2], )
            function = lambda params: forward_backward_prop(data,
                                                            labels,
                                                            params,
                                                            dimensions)
            result = gradcheck_naive(function, params)
            self.assertTrue(np.sum(result) <= len(params) * 1e-5)
            end = time.time()
            print("Test time = {:f}(s)\n".format(end - start))

    def test_dimensions(self):
        random_data_sizes = np.random.randint(1, 100, 5)
        random_Dx = np.random.randint(2, 20, 5)
        random_H = np.random.randint(2, 20, 5)
        random_Dy = np.random.randint(2, 20, 5)
        for i in range(5):
            start = time.time()
            N = random_data_sizes[i]
            dimensions = [random_Dx[i], random_H[i], random_Dy[i]]
            print("data size = {}".format(N))
            print("dimensions = {}".format(dimensions))
            data = np.random.randn(N, dimensions[0])
            labels = np.zeros((N, dimensions[2]))
            for i in range(N):
                labels[i, random.randint(0, dimensions[2]-1)] = 1
            params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
                dimensions[1] + 1) * dimensions[2], )
            function = lambda params: forward_backward_prop(data,
                                                            labels,
                                                            params,
                                                            dimensions)
            result = gradcheck_naive(function, params)
            self.assertTrue(np.sum(result) <= len(params) * 1e-5)
            end = time.time()
            print("Test time = {:f}(s)\n".format(end - start))


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
    python q2_neural.py This function will not be called by
    the autograder, nor will your additional tests be graded.
    """
    print("Running your sanity checks...")
    # ## YOUR CODE HERE
    suite = unittest.TestSuite()
    for method in dir(TestNN):
        if method.startswith("test"):
            suite.addTest(TestNN(method))
    unittest.TextTestRunner().run(suite)
    # ## END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
