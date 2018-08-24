import matplotlib.pyplot as plt#
import numpy as np#
import sklearn#
import os
import glob
import librosa
import pandas as pd
from IPython.display import Audio
from sklearn.externals import joblib#


def zero_rule(vec_labels):
    # Now lets predict the labels of the test data!
    ones = np.ones(len(vec_labels))
    # We can use sklearn to compute the accuracy score
    accuracy = sklearn.metrics.accuracy_score(vec_labels, ones)
    return accuracy


def load_cls_predict(filename, vec_features):
    # Load trained model (RF)
    filename = 'finalized_model_RF_1000_VGGish.sav' 
    # load the model from disk
    clf = joblib.load(filename)

    # Now lets predict the labels of the test data!
    predictions = clf.predict(test_features)
    # We can use sklearn to compute the accuracy score
    accuracy = sklearn.metrics.accuracy_score(test_labels, predictions)
    return (predictions, accuracy)


def load_scaler_transform(filename, vec_features):
    # Load scaler (SVM)
    filename = '../scaler_VGGish.sav' 
    # load the model from disk
    scaler = joblib.load(filename)
    # Transform data
    return scaler.transform(test_features)


def confusion_matrix(vec_labels, predictions):
    # lets compute the show the confusion matrix:
    cm = sklearn.metrics.confusion_matrix(vec_labels, predictions)
    
    return cm


def plot_confusion_matrix(cm, labels): # labels é [absent, present] 
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap='gray')
    for i, line in enumerate(cm):
        for j, l in enumerate(line):
            ax.text(j, i, l, size=20, color='green')
    ax.set_xticks(range(len(cm)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(cm)))
    ax.set_yticklabels(labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()
    
    

if __name__ == "__main__":
    
    print ("I am a function import file =D")
