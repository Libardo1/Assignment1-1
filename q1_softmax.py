import numpy as np
import random
import unittest
from math import isnan


# fuction to normalize an array
def normalize(arr):
    size = len(arr)
    sum_all = np.sum(arr)
    y = np.ndarray(shape=(size), dtype=float)
    for i in range(size):
        y[i] = arr[i]/sum_all
    return y


def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the
    written assignment!
    """

    # ## YOUR CODE HERE

    # when x is an one dimensional input
    # we transform it in an array having
    # x as its only mem
    if type(x[0]) != np.ndarray:
        x = x.reshape((1, len(x)))

    nrows = len(x)
    ncols = len(x[0])
    y = np.ndarray(shape=(nrows, ncols), dtype=float)
    for i, row in enumerate(x):

        # when the array has a high std we normalize the array
        if np.std(row) >= 290:
            row = normalize(row)

        stability_constant = np.max(row)
        y[i] = np.exp(row - stability_constant)/
        np.sum(np.exp(row - stability_constant))

    # ## END YOUR CODE
    return y


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1, 2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001, -1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print("You should verify these results!\n")


class TestSoftmax(unittest.TestCase):

    # Here we test if the result of the softmax function
    # lies in the interval [0,1].
    # One thing, in some tests we have:
    # np.sum(softmax(y)) = 1.0000000000000002
    # I don't know if this is a problem or not. So I change
    def test_upperbound(self):
        for k in range(300):
            y = np.ndarray(shape=(10), dtype=float)
            for i in range(10):
                y[i] = random.randint(-100, 100)
            sum_test = np.sum(softmax(y))
            self.assertTrue(sum_test <= 1.0001,
                            """\n if y = {0} \n, then np.sum(softmax(y))
                            = {1} is bigger than 1"""
                            .format(y, sum_test))

    # Here we test if the softmax function can handle
    # arrays with high std like [1001,1,1,1]
    def test_high_std(self):
        for k in range(300):
            y = np.ndarray(shape=(10), dtype=float)
            for i in range(0, 9):
                y[i] = random.randint(1, 3)
            y[9] = random.randint(1001, 1100)
            self.assertFalse(isnan(np.sum(softmax(y))),
                             """\n if y = {0} \n, then std = {1}
                             and the softmax will not work"""
                             .format(y, np.std(y)))

    # Here we test if the softmax function can handle
    # arrays with big and and low numbers
    def test_high_low(self):
        for k in range(300):
            y = np.ndarray(shape=(2, 2), dtype=float)
            y[0][0] = random.randint(-20000, -10000)
            y[0][1] = y[0][0]+1
            y[1][0] = random.randint(10000, 20000)
            y[1][1] = y[1][0]+1
            self.assertTrue(np.amax(np.fabs(softmax(y) - np.array(
             [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6,
                            """\n if y = {0} \n,
                            then softmax is {1}: problem"""
                            .format(y, softmax(y)))


def test_softmax():
    """
    Use this space to test your softmax
    implementation by running: python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    # ## YOUR CODE HERE
    suite = unittest.TestSuite()
    for method in dir(TestSoftmax):
        if method.startswith("test"):
            suite.addTest(TestSoftmax(method))
    unittest.TextTestRunner().run(suite)
    # ## END YOUR CODE

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
