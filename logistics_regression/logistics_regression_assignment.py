import nltk
from os import getcwd
import numpy as np
from nltk.corpus import twitter_samples

from utils.utils import build_freqs, process_tweet

# Use numpy.squeeze() to make an (m,1) dimensional array into an (m,) array

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

def prepare_data():
    all_positive_tweets = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets = twitter_samples.strings('negative_tweets.json')
    test_pos = all_positive_tweets[4000:]
    train_pos = all_positive_tweets[:4000]
    test_neg = all_negative_tweets[4000:]
    train_neg = all_negative_tweets[:4000]
    test_x = test_pos + test_neg
    train_x = train_pos + train_neg

    test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)
    train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    freqs = build_freqs(train_x, train_y)

    return test_x, train_x, test_y, train_y, freqs


def sigmoid(z):
    '''
        Input:
            z: is the input (can be a scalar or an array)
        Output:
            h: the sigmoid of z
    '''
    return 1 / (1 + np.exp(-z))


def test_sigmoid():
    if (sigmoid(0) == 0.5):
        print('SUCCESS!')
    else:
        print('Oops!')

    if (sigmoid(4.92) == 0.9927537604041685):
        print('CORRECT!')
    else:
        print('Oops again!')

    if all(sigmoid(np.asarray([0, 4.92])) == np.asarray([0.5, 0.9927537604041685])):
        print('CORRECT!')


# UNQ_C2 GRADED FUNCTION: gradientDescent
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = -(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h))) / m

        # update the weights theta
        theta = theta - alpha * (np.dot(x.T, (h - y))) / m

    ### END CODE HERE ###
    J = float(J)
    return J, theta


def test_gradient_descent():
    # Check the function
    # Construct a synthetic test case using numpy PRNG functions
    np.random.seed(1)
    # X input is 10 x 3 with ones for the bias terms
    tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
    # Y Labels are 10 x 1
    tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

    # Apply gradient descent
    tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
    print(f"The cost after training is {tmp_J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")


def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    ### START CODE HERE ###

    # loop through each word in the list of words
    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0), 0)

    ### END CODE HERE ###
    assert (x.shape == (1, 3))
    return x

def test_extract_features():
    test_x, train_x, test_y, train_y, freqs = prepare_data()
    tmp1 = extract_features(train_x[0], freqs)
    print(tmp1)


def train_model():
    test_x, train_x, test_y, train_y, freqs = prepare_data()
    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # Apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
    return J, theta


# UNQ_C4 GRADED FUNCTION: predict_tweet
def predict_tweet(tweet, freqs, theta):
    '''
    Input:
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output:
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE ###

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    ### END CODE HERE ###

    return y_pred


# UNQ_C5 GRADED FUNCTION: test_logistic_regression
def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    ### START CODE HERE ###

    # the list for storing predictions
    y_hat = list()

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    y_hat = np.asarray(y_hat)
    test_y = np.squeeze(test_y)
    accuracy = np.sum(y_hat == test_y) / test_y.shape[0]

    ### END CODE HERE ###

    return accuracy

if __name__ == '__main__':
    J, theta = train_model()
    test_x, _, test_y, _, freqs = prepare_data()
    # Test the accuracy on unseen data
    tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
    print('test accuracy is ', tmp_accuracy)
    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great',
                  'great great great', 'great great great great']:
        print('%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))