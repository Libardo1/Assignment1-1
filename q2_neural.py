import numpy as np
import random

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
    def fix_shape(array):
        return np.reshape(array, (array.shape[0], array.shape[2]))

    all_u = fix_shape(np.array([(x.dot(W1) + b1) for x in data]))
    all_h = sigmoid(all_u)
    all_theta = fix_shape(np.array([(h.dot(W2) + b2) for h in all_h]))
    all_y_hat = fix_shape(np.array([softmax(theta) for theta in all_theta]))
    all_costs = np.array([-np.sum(y*np.log(y_hat))
                         for y, y_hat in zip(labels, all_y_hat)])
    cost = np.mean(all_costs)
    # ## END YOUR CODE

    # ## YOUR CODE HERE: backward propagation
    def del_cost_del_W1(j, i):
        e_i = np.sum((all_y_hat - labels)*W2[i], 1)
        sigmoid_u = sigmoid_grad(sigmoid(all_u.T[i]))
        x_j = data.T[j]
        return np.mean(e_i*sigmoid_u*x_j)

    def del_cost_del_b1(j, i):
        W2_i = W2[i]
        u_i = sigmoid_grad(sigmoid(all_u.T[i]))
        subtraction = (all_y_hat - labels)*W2_i
        multiplication = [vector*scalar
                          for vector, scalar in zip(subtraction, u_i)]
        result = np.array(multiplication)
        result = np.mean(np.sum(result, 1))
        return result

    def del_cost_del_W2(i, j):
        result = [(Y[j] - y[j])*h[i]
                  for Y, y, h in zip(all_y_hat, labels, all_h)]
        return np.mean(result)

    def del_cost_del_b2(i, j):
        subtraction_j = all_y_hat.T[j] - labels.T[j]
        return np.mean(subtraction_j)

    def get_grad(array, grad_function):
        matrix = np.array(array, copy=True)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                matrix[i][j] = grad_function(i, j)
        return matrix

    gradW1 = get_grad(W1, del_cost_del_W1)
    gradb1 = get_grad(b1, del_cost_del_b1)
    gradW2 = get_grad(W2, del_cost_del_W2)
    gradb2 = get_grad(b2, del_cost_del_b2)
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
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
                                                         dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
    python q2_neural.py This function will not be called by
    the autograder, nor will your additional tests be graded.
    """
    print("Running your sanity checks...")
    # ## YOUR CODE HERE
    print("Sex is better than machine learning")
    # ## END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
