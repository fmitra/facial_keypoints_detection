""" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

https://www.kaggle.com/c/facial-keypoints-detection

Competition goal is to predict keypoint positions on face images.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import learning_curve
from custom_net import save_submission, load_data, TRAIN, TEST, FLOOKUP


def train(X, y):
    model = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='sgd',
        alpha=0.001,
        learning_rate='invscaling',
        max_iter=200,
        random_state=0,
        verbose=True
        )

    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, n_jobs=4)

    plt.figure()
    plt.title("Learning Curves")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    print("train {}".format(train_scores_mean))
    print("valid {}".format(valid_scores_mean))

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    # plt.show()

    model.fit(X, y)
    return model

def predict(model, X):
    y = model.predict(X)
    save_submission(y)

def start():
    X, x, y = load_data()
    model = train(X, y)
    predict(model, x)

if __name__ == '__main__':
    start()




