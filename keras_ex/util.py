# @Author: Tang Yubin <tangyubin>
# @Date:   2019-05-26T12:04:11+08:00
# @Email:  tang-yu-bin@qq.com
# @Last modified by:   tangyubin
# @Last modified time: 2019-05-26T16:24:35+08:00

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    precision_score = precision(y_true, y_pred)
    recall_score = recall(y_true, y_pred)
    return 2 * ((precision_score * recall_score) / (precision_score + recall_score))

class ParamError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def plot_learning_curve(x, y, title = "Learning Curve",
                      xlabel = "Epochs", ylabel="Loss",
                      ylim = None,
                      labels = None, colors = None):
    """
    Generate a simple plot of learning curve.
    Parameters
    --------------
    x: array-like, shape (n_epochs,)
    y: array-like, or list of array-like, shape (n_epochs,) or (n_scores, n_epochs)
    """
    check_params = {"labels":labels, "colors":colors}
    for k, v in check_params.items():
        if v != None and (type(v) is not list or len(v) != len(y)):
            raise ParamError("param {} must have same length with y".format(k))

    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if type(y) is list:
        for i, yy in enumerate(y):
            color = colors[i] if colors is not None else 'r'
            label = labels[i] if labels is not None else 'Score {}'.format(i)
            plt.plot(x, yy, 'o-', color=color, label=label)
    else:
        plt.plot(x, y, 'o-', color='r', label=ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(loc='best')
    plt.show()
