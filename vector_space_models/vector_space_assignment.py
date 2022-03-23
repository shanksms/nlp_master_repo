# Run this cell to import packages.
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
Now you will write a function that will use the word embeddings to predict relationships among words.

The function will take as input three words.
The first two are related to each other.
It will predict a 4th word which is related to the third word in a similar manner as the two first words are related to
each other.
As an example, "Athens is to Greece as Bangkok is to __"?
You will write a program that is capable of finding the fourth word.
We will give you a hint to show you how to compute this.
"""


def cosine_similarity(a, b):
    '''
    𝑐𝑜𝑠(𝜃) = a.b / ||a|| * ||b||
    a.b is vector dot product.
    ||a|| and ||b|| are norms of a and b vectors.
    𝐴  and  𝐵  represent the word vectors and  𝐴𝑖  or  𝐵𝑖  represent index i of that vector. Note that if A and B
    are identical, you will get  𝑐𝑜𝑠(𝜃)=1 .

    Otherwise, if they are the total opposite, meaning,  𝐴=−𝐵 , then you would get  𝑐𝑜𝑠(𝜃)=−1 .
    If you get  𝑐𝑜𝑠(𝜃)=0 , that means that they are orthogonal (or perpendicular).
    Numbers between 0 and 1 indicate a similarity score.
    Numbers between -1 and 0 indicate a dissimilarity score.
        Input:
            A: a numpy array which corresponds to a word vector
            B: A numpy array which corresponds to a word vector
        Output:
            cos: numerical number representing the cosine similarity between A and B.
    '''
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def euclidean(a, b):
    """
       Input:
           A: a numpy array which corresponds to a word vector
           B: A numpy array which corresponds to a word vector
       Output:
           d: numerical number representing the Euclidean distance between A and B.
       """
    d = np.linalg.norm(a - b)
    return d


"""
Now, you will use the previous functions to compute similarities between vectors, and use these to find the capital
 cities of countries. You will write a function that takes in three words, and the embeddings dictionary. Your task is to find the capital cities. For example, given the following words:

1: Athens 2: Greece 3: Baghdad,
your task is to predict the country 4: Iraq.

Instructions:

To predict the capital you might want to look at the King - Man + Woman = Queen example above, and implement that scheme
into a mathematical function, using the word embeddings and a similarity function.

Iterate over the embeddings dictionary and compute the cosine similarity score between your vector and the current word
embedding.

You should add a check to make sure that the word you return is not any of the words that you fed into your function. 
Return the one with the highest score.
"""


def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    """
        Input:
            city1: a string (the capital city of country1)
            country1: a string (the country of capital1)
            city2: a string (the capital city of country2)
            embeddings: a dictionary where the keys are words and
        Output:
            countries: a dictionary with the most likely country and its similarity score
    """
    group = {city1, country1, city2}
    city1_vector = embeddings[city1]
    country1_vector = embeddings[country1]
    city2_vector = embeddings[city2]
    country2_vector = country1_vector - city1_vector + city2_vector
    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''
    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():

        # first check that the word is not already in the 'group'
        if word not in group:

            # get the word embedding
            word_emb = embeddings[word]

            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity(country2_vector, word_emb)

            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:
                # update the similarity to the new, better similarity
                similarity = cur_similarity

                # store the country as a tuple, which contains the word and the similarity
                country = (word, similarity)

    ### END CODE HERE ###

    return country



def main():
    base_path = Path(__file__).parent

    data = pd.read_csv(os.path.join(base_path, 'capitals.txt'), delimiter=' ')
    data.columns = ['city1', 'country1', 'city2', 'country2']

    # print first five elements in the DataFrame
    data.head(5)
    word_embeddings = pickle.load(open(os.path.join(base_path, 'word_embeddings_subset.p'), "rb"))
    len(word_embeddings)  # there should be 243 words that will be used in this assignment
    print("dimension: {}".format(word_embeddings['Spain'].shape[0]))

    # test cosine similarity
    king = word_embeddings['king']
    queen = word_embeddings['queen']

    print('cosine similarity', cosine_similarity(king, queen))
    print('euclidian distance', euclidean(king, queen))

    print('Country of Cairo', get_country('Athens', 'Greece', 'Cairo', word_embeddings))


if __name__ == '__main__':
    main()
