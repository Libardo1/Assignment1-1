import numpy as np
import unittest


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """

    # ## YOUR CODE HERE
    x = 1/(1 + np.exp(-x))
    # ## END YOUR CODE
    return x


def sigmoid_grad(f):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input f should be the sigmoid
    function value of your original input x.
    """

    # ## YOUR CODE HERE
    f = f*(1-f)
    # ## END YOUR CODE

    return f


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    assert np.amax(f - np.array([[0.73105858, 0.88079708],
                   [0.26894142, 0.11920292]])) <= 1e-6
    print(g)
    assert np.amax(g - np.array([[0.19661193, 0.10499359],
                   [0.19661193, 0.10499359]])) <= 1e-6
    print("You should verify these results!\n")


class TestSigmoid(unittest.TestCase):

    def test_upperbound(self):
        above_zero_ints = np.random.randint(1, 100, 300)
        result = sigmoid(above_zero_ints)
        result = np.sum(result)
        self.assertTrue(result <= 300,
                        """There is some number above 1,
                        hence result = {} and not a number
                        less or equal to 300 """.format(result))

    def test_turning_point(self):
        result = sigmoid(0)
        self.assertTrue(result == 0.5,
                        """result = {} and not
                        0.5""".format(result))

    def test_lowerbound(self):
        below_zero_ints = np.random.randint(-100, -1, 300)
        result = sigmoid(below_zero_ints)
        result = np.sum(result)
        self.assertTrue(result < 150,
                        """There is some number above or equal to 0.5,
                        hence result = {} and not a number
                        less to 150 """.format(result))


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
    python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    # ## YOUR CODE HERE
    suite = unittest.TestSuite()
    for method in dir(TestSigmoid):
        if method.startswith("test"):
            suite.addTest(TestSigmoid(method))
    unittest.TextTestRunner().run(suite)
    # ## END YOUR CODE

if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
