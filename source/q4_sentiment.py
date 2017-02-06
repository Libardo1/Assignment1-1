import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
import os
from datetime import datetime, timedelta
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cs224d.data_utils import *

from q3_sgd import load_saved_params, sgd
from q4_softmaxreg import softmaxRegression,\
 getSentenceFeature, accuracy, softmax_wrapper

start = time.time()

# Try different regularizations and pick the best!
# NOTE: fill in one more "your code here" below before running!
REGULARIZATION = None   # Assign a list of floats in the block below
# ## YOUR CODE HERE
# best_so_far = 8.410683E-02
# reg0 = np.array([best_so_far, 3.320813E-02])
# reg_plus = np.random.random_sample([2]) / 100
# reg_plus += best_so_far
# reg_minus = - np.random.random_sample([2]) / 100
# reg_minus += best_so_far
# reg2 = np.random.random_sample([1]) / 50
# reg3 = np.random.random_sample([1]) / 1000
# REGULARIZATION = np.concatenate((reg0, reg_plus, reg_minus))
REGULARIZATION = [0.0,
                  0.00001,
                  0.00003,
                  0.0001,
                  0.0003,
                  0.001,
                  0.003,
                  0.03,
                  0.01]
REGULARIZATION.sort()
print("All the regularization params are = {}".format(REGULARIZATION))
# ## END YOUR CODE

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# Load the word vectors we trained earlier
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Try our regularization parameters
results = []
for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization)

    training_steps = 1

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures,
                                                  trainLabels,
                                                  weights,
                                                  regularization),
                  weights,
                  3,
                  training_steps,
                  PRINT_EVERY=100)

    # Test on train set
    _, _, pred = softmaxRegression(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = softmaxRegression(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)

    # Save the results and weights
    results.append({
        "reg": regularization,
        "weights": weights,
        "train": trainAccuracy,
        "dev": devAccuracy})

# Print the accuracies
print("")
print("=== Recap ===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (
        result["reg"],
        result["train"],
        result["dev"]))
print("")

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None

# ## YOUR CODE HERE
best_result = - float('inf')
for i in range(len(results)):
    if results[i]["dev"] > best_result:
        best_result = results[i]["dev"]
        BEST_REGULARIZATION = results[i]["reg"]
        BEST_WEIGHTS = results[i]["weights"]

# ## END YOUR CODE

# Test your findings on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmaxRegression(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value: %E" % BEST_REGULARIZATION)
print("Test accuracy (%%): %f" % accuracy(testLabels, pred))

# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("q4_reg_v_acc.png")
# plt.show()

# Getting the duration of the process
end = time.time()
num_steps = training_steps*len(REGULARIZATION)
general_duration = end - start
sec = timedelta(seconds=int(general_duration))
d_time = datetime(1, 1, 1) + sec
print('The duration of the whole training with % s steps is %.2f seconds,' % (num_steps, general_duration))
print "which is equal to:  %d:%d:%d:%d" % (d_time.day-1, d_time.hour, d_time.minute, d_time.second),
print(" (DAYS:HOURS:MIN:SEC)")

# Sending an email with the results
cwd = os.getcwd()

script_name = "q4_sentiment.py"

fromaddr = "robotanara@gmail.com"
toaddr = "felipessalvador@gmail.com"
msg = MIMEMultipart()
msg['From'] = fromaddr
msg['To'] = toaddr
msg['Subject'] = "End of {}".format(script_name)

body = """Dear Felipe,
the script {} is done.
A review of the process can be seen in the following attachments.

Best,
Nara""".format(script_name)
msg.attach(MIMEText(body, 'plain'))

filename1 = "nohup.out"
attachment1 = open(cwd + '/' + filename1, "rb")

part1 = MIMEBase('application', 'octet-stream')
part1.set_payload((attachment1).read())
encoders.encode_base64(part1)
part1.add_header('Content-Disposition', "attachment; filename= %s" % filename1)

msg.attach(part1)

filename2 = "q4_reg_v_acc.png"
attachment2 = open(cwd + '/' + filename2, "rb")

part2 = MIMEBase('application', 'octet-stream')
part2.set_payload((attachment2).read())
encoders.encode_base64(part2)
part2.add_header('Content-Disposition', "attachment; filename= %s" % filename2)

msg.attach(part2)

password = os.environ.get('MY_ENV_VAR')

server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(fromaddr, password)
text = msg.as_string()
server.sendmail(fromaddr, toaddr, text)
server.quit()
