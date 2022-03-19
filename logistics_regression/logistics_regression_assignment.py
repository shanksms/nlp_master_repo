import nltk
from os import getcwd
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples

from logistics_regression.build_frequency import build_freqs, process_tweet



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

    return test_x, train_x, test_y, train_y


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


if __name__ == '__main__':
    #prepare_data()
    test_sigmoid()
