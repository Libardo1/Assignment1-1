import numpy as np
import random
import unittest
from q2_sigmoid import sigmoid, sigmoid_grad

# First implement a gradient checker by filling in the following functions


def gradcheck_naive(f, x):
    """
    Gradient check for a function f
    - f should be a function that takes a single argument
    and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    # Evaluate function value at original point
    fx, grad = f(x)
    h = 1e-4
    test = True

    """
     --- about nd.nditer ---
    flags = multi_index causes a multi-index, or a tuple of indices
    with one per iteration dimension, to be tracked.
    op_flags : list of list of str, optional
    this is a list of flags for each operand. At minimum, one of readonly,
    readwrite, or writeonly must be specified.
    readwrite indicates the operand will be read from and written to.
    """

    # Iterate over all indexes in x
    all_dif = np.array(x, copy=True)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        """ try modifying x[ix] with h defined above to compute numerical gradients
        make sure you call random.setstate(rndstate) before calling f(x)
        each time, this will make it possible to test cost functions with
        built in randomness later"""
        # YOUR CODE HERE:
        x_plus_h = np.array(x, copy=True)
        x_plus_h[ix] = x_plus_h[ix] + h
        random.setstate(rndstate)
        fxh_plus, _ = f(x_plus_h)
        x_minus_h = np.array(x, copy=True)
        x_minus_h[ix] = x_minus_h[ix] - h
        random.setstate(rndstate)
        fxh_minus, _ = f(x_minus_h)
        numgrad = (fxh_plus - fxh_minus)/(2*h)
        # END YOUR CODE

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        all_dif[ix] = reldiff
        if reldiff > 1e-5:
            test = False
            string = """
            Your gradient = {0}
            Numerical gradient = {1}""".format(grad[ix], numgrad)
            print(str(ix) + ": " + string)
            print("fx ={}".format(fx))
            print("fxh_plus ={}".format(fxh_plus))
            print("fxh_minus ={}".format(fxh_minus))
        # For debugging with a bunch of params is
        # useful to add the following:

        else:
            print(str(ix) + ": OK")

        # Step to next dimension
        it.iternext()
    if test:
        print("Gradient check passed!")

    # add this return to test
    return all_dif


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))   # 2-D test
    print("")


class TestGrad(unittest.TestCase):

    def test_sigmoid(self):
        sig = lambda x: (sigmoid(x), sigmoid_grad(sigmoid(x)))

        random_ints = np.random.randint(1, 100, 100)
        random_floats = np.random.random_sample((100,))
        random_floats = random_ints*random_floats
        for number in random_floats:
            result = gradcheck_naive(sig, np.array(number))
            self.assertTrue(float(result) <= 1e-5)

    def test_polynomial1(self):
        poly = lambda x: (x[0]**2 + x[1]**2,
                          np.array([2*x[0], 2*x[1]]))

        random_ints1 = np.random.randint(1, 100, 100)
        random_ints2 = np.random.randint(1, 100, 100)
        random_floats = np.random.random_sample((100,))
        random_floats1 = random_ints1*random_floats
        random_floats2 = random_ints2*random_floats
        tuples = np.array(zip(random_floats1, random_floats2))
        for tuple in tuples:
            result = gradcheck_naive(poly, tuple)
            self.assertTrue(np.sum(result) <= 2*1e-5)

    def test_polynomial2(self):
        poly = lambda x: (x[0]**2 + x[1]**3 + x[2]**4,
                          np.array([2*x[0], 3*x[1]**2, 4*x[2]**3]))

        random_ints1 = np.random.randint(1, 100, 100)
        random_ints2 = np.random.randint(1, 100, 100)
        random_ints3 = np.random.randint(-100, -10, 100)
        random_floats = np.random.random_sample((100,))
        random_floats1 = random_ints1*random_floats
        random_floats2 = random_ints2*random_floats
        random_floats3 = random_ints3*random_floats
        triples = np.array(zip(random_floats1, random_floats2, random_floats3))
        for triple in triples:
            result = gradcheck_naive(poly, triple)
            self.assertTrue(np.sum(result) <= 3*1e-5)

    def test_polynomial3(self):
        """
        My test do not pass if I use np.exp() in the pol
        """
        poly = lambda x: (x[0]**2 + x[1]**3 - 8*np.log(x[2]) + np.log(x[3]),
                          np.array([2*x[0], 3*x[1]**2, -8/x[2], 1/x[3]]))

        random_ints1 = np.random.randint(1, 100, 100)
        random_ints2 = np.random.randint(1, 100, 100)
        random_ints3 = np.random.randint(1, 100, 100)
        random_ints4 = np.random.randint(1, 100, 100)
        random_floats = np.random.random_sample((100,))
        random_floats1 = random_ints1*random_floats
        random_floats2 = random_ints2*random_floats
        random_floats3 = random_ints3*random_floats
        random_floats4 = random_ints4*random_floats
        quadruples = np.array(zip(random_floats1,
                                  random_floats2,
                                  random_floats3,
                                  random_floats4))
        for quadruple in quadruples:
            result = gradcheck_naive(poly, quadruple)
            self.assertTrue(np.sum(result) <= 4*1e-5)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    # ## YOUR CODE HERE
    suite = unittest.TestSuite()
    for method in dir(TestGrad):
        if method.startswith("test"):
            suite.addTest(TestGrad(method))
    unittest.TextTestRunner().run(suite)
    # ## END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
