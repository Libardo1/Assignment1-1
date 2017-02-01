import numpy as np
import random
from collections import Counter
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each
    # row of a matrix to have unit length

    # ## YOUR CODE HERE
    all_norm2 = np.sqrt(np.sum(np.power(x, 2), 1))
    all_norm2 = 1/all_norm2
    x = x * all_norm2[:, np.newaxis]
    # ## END YOUR CODE

    return x


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    assert(x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print(" ")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    # ## YOUR CODE HERE
    y_hat = (softmax(outputVectors.dot(predicted))).flatten()
    y = np.zeros(outputVectors.shape[0])
    y[target] = 1
    cost = np.sum(y * np.log(y_hat)) * -1
    subtraction = y_hat - y
    gradPred = np.sum(subtraction*outputVectors.T, 1)
    grad = np.outer(subtraction, predicted)
    # ## END YOUR CODE

    return cost, gradPred, grad


def negSamplingCostAndGradient(predicted,
                               target,
                               outputVectors,
                               dataset,
                               K=10):

    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.

    # Note: See test_word2vec below for dataset's initialization.

    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    # ## YOUR CODE HERE
    random_sample = []
    while len(random_sample) < K:
        pick_idx = dataset.sampleTokenIdx()
        if pick_idx != target:
            random_sample.append(pick_idx)
    sample_vectors = outputVectors[random_sample, :]
    target_pred = outputVectors[target].dot(predicted)
    sample_pred = sample_vectors.dot(predicted)
    cost = - (np.log(sigmoid(target_pred)) +
              np.sum(np.log(sigmoid(-sample_pred))))

    gradPred = - sigmoid(- target_pred)*outputVectors[target] + np.dot(
        sigmoid(sample_pred), sample_vectors)

    grad = np.zeros(outputVectors.shape)
    grad[target] = - sigmoid(- target_pred) * predicted
    counter = Counter(random_sample)
    for i in counter.keys():
        grad[i] = counter[i]*(sigmoid(outputVectors[i].dot(predicted)) *
                              predicted)

    # ## END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord,
             C,
             contextWords,
             tokens,
             inputVectors,
             outputVectors,
             dataset,
             word2vecCostAndGradient=softmaxCostAndGradient):

    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    # ## YOUR CODE HERE
    current_index = tokens[currentWord]
    v_hat = inputVectors[current_index]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        target = tokens[word]
        word_cost, word_gradPred, word_grad = word2vecCostAndGradient(v_hat,
                                                                      target,
                                                                      outputVectors,
                                                                      dataset)
        cost += word_cost
        gradIn[current_index] += word_gradPred
        gradOut += word_grad
    # ## END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord,
         C,
         contextWords,
         tokens,
         inputVectors,
         outputVectors,
         dataset,
         word2vecCostAndGradient=softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    # ## YOUR CODE HERE
    current_index = tokens[currentWord]
    v_hat = inputVectors[current_index]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for word in contextWords:
        target = tokens[word]
        word_cost, word_gradPred, word_grad = word2vecCostAndGradient(v_hat,
                                                                      target,
                                                                      outputVectors,
                                                                      dataset)
        cost += word_cost
        gradIn[current_index] += word_gradPred
        gradOut += word_grad
    # ## END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################


def word2vec_sgd_wrapper(word2vecModel,
                         tokens,
                         wordVectors,
                         dataset,
                         C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2, :]
    outputVectors = wordVectors[N/2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword,
                                     C1,
                                     context,
                                     tokens,
                                     inputVectors,
                                     outputVectors,
                                     dataset,
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        center_word = tokens[random.randint(0, 4)]
        context = [tokens[random.randint(0, 4)] for i in xrange(2*C)]
        return center_word, context

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram,
                                                     dummy_tokens,
                                                     vec,
                                                     dataset,
                                                     5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram,
                                                     dummy_tokens,
                                                     vec,
                                                     dataset,
                                                     5,
                                                     negSamplingCostAndGradient), dummy_vectors)
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow,
    #                                                  dummy_tokens,
    #                                                  vec,
    #                                                  dataset,
    #                                                  5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow,
    #                                                  dummy_tokens,
    #                                                  vec,
    #                                                  dataset,
    #                                                  5,
    #                                                  negSamplingCostAndGradient), dummy_vectors)

    # print "\n=== Results ==="
    # result1 = skipgram("c",
    #                        3,
    #                        ["a", "b", "e", "d", "b", "c"],
    #                        dummy_tokens,
    #                        dummy_vectors[:5, :],
    #                        dummy_vectors[5:, :],
    #                        dataset)

    # result2 = skipgram1("c",
    #                        3,
    #                        ["a", "b", "e", "d", "b", "c"],
    #                        dummy_tokens,
    #                        dummy_vectors[:5, :],
    #                        dummy_vectors[5:, :],
    #                        dataset)
    # result2 = skipgram("c",
    #                    1,
    #                    ["a", "b"],
    #                    dummy_tokens,
    #                    dummy_vectors[:5, :],
    #                    dummy_vectors[5:, :],
    #                    dataset,
    #                    negSamplingCostAndGradient)

    # result3 = cbow("a",
    #                2,
    #                ["a", "b", "c", "a"],
    #                dummy_tokens,
    #                dummy_vectors[:5, :],
    #                dummy_vectors[5:, :],
    #                dataset)

    # result4 = cbow("a",
    #                2,
    #                ["a", "b", "a", "c"],
    #                dummy_tokens,
    #                dummy_vectors[:5, :],
    #                dummy_vectors[5:, :],
    #                dataset,
    #                negSamplingCostAndGradient)

    # print(result1[1])
    # print(result1[2])
    # print(result2[0])
    # print(result2[1])
    # print(result2[2])
    # print(result3)
    # print(result4)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
