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


"""
Now you will test your new function on the dataset and check the accuracy of the model:

Accuracy=Correct # of predictions / Total # of predictions
 
Instructions: Write a program that can compute the accuracy on the dataset provided for you. You have to iterate over 
every row to get the corresponding words and feed them into you get_country function above.
"""
def get_accuracy(word_embeddings, data, get_country=get_country):

    correct_count = 0
    for _, row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']
        predicted_country, _ = get_country(city1, country1, city2, word_embeddings)
        if predicted_country == country2:
            correct_count += 1

    return correct_count / len(data)


"""
You will write a program that takes in a data set where each row corresponds to a word vector.

The word vectors are of dimension 300.

Use PCA to change the 300 dimensions to n_components dimensions.

The new matrix should be of dimension m, n_componentns.

First de-mean the data

Get the eigenvalues using linalg.eigh. Use 'eigh' rather than 'eig' since R is symmetric. The performance gain when using eigh instead of eig is substantial.

Sort the eigenvectors and eigenvalues by decreasing order of the eigenvalues.

Get a subset of the eigenvectors (choose how many principle components you want to use using n_components).

Return the new transformation of the data by multiplying the eigenvectors with the original data.
"""
def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    ### START CODE HERE ###
    # mean center the data
    X_demeaned = X - np.mean(X, axis=0)
    #.reshape((X.shape[0], 1))

    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')
    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)

    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted[:, 0:n_components]

    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T

    ### END CODE HERE ###

    return X_reduced


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

    accuracy = get_accuracy(word_embeddings, data)
    print(f"Accuracy is {accuracy:.2f}")

    #########################PCA testing############
    np.random.seed(1)
    X = np.random.rand(3, 10)
    X_reduced = compute_pca(X, n_components=2)
    print("Your original matrix was " + str(X.shape) + " and it became:")
    print(X_reduced)




if __name__ == '__main__':
    main()
