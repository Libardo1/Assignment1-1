from datetime import datetime, timedelta
import time
import random
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cs224d.data_utils import *
from q3_word2vec import *
from q3_sgd import *

start = time.time()
# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# number of traing steps
num_steps = 15

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
	dimVectors, np.zeros((nWords, dimVectors))), axis=0)
wordVectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, 
    	negSamplingCostAndGradient), 
    wordVectors, 0.3, num_steps, None, True, PRINT_EVERY=10)
print "sanity check: cost at convergence should be around or below 10"

# sum the input and output word vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", 
	"good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
	"worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", 
	"annoying"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2]) 

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], 
    	bbox=dict(facecolor='green', alpha=0.1))
    
plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
# plt.show()
end = time.time()

general_duration = end - start
sec = timedelta(seconds=int(general_duration))
d_time = datetime(1, 1, 1) + sec
print('The duration of the whole training with % s steps is %.2f seconds,' % (num_steps, general_duration))
print "which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second),
print(" (DAYS:HOURS:MIN:SEC)")
