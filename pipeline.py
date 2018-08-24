from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import librosa
import glob
import json
import csv
import os

from paths import *
from metrics import *
""" 
    Important constants 
"""

# Specify the labels (classes) we're going to classify the data into
label0 = 'abscent'
label1 = 'present'
labels = [label0, label1]

# Make 1 second summarization as features with half second of hop length
# Each frame has 10 ms
feature_length = 96
half_sec = 48

""""""
            
            
# Load musics from train set
def load_features(files, feature):
    '''Get a list of files.  Returns the loaded features and labels for the files.

        Parameters
        ----------           
        files: list
            Files to load the features
        
        feature: string
            Type of feature to be loaded

        Returns
        -------
        load_features : list
            A list with features from files
            
        load_labels : list
            A list with labels from files
        '''
    
    # Define lists to store the training features and corresponding training labels
    load_features = []
    load_labels = []
    
    
    # For every audio file in the set, load the feature file and then save the features
    # and corresponding label in the designated lists
    for file in files:
        tf = os.environ["FEATURE_PATH"]+file+"/"+file+"_"+feature+".csv"
        tl = os.environ["FEATURE_PATH"]+file+"/"+file+"_vocal.csv"
        
        print (tf)

        print("filename: {:s}".format(os.path.basename(tf)))

        # Load VGGish audio embeddings
        vggish = pd.read_csv(tf, index_col=None, header=None) ### IF NOT EXIST, call the calculate embeddings... =D
        vggish = pd.DataFrame.as_matrix(vggish)

        # Read labels for each frame
        activation = pd.read_csv(tl, index_col=None, header=None)
        activation = pd.DataFrame.as_matrix(activation)
        activation = activation.T[0][48:] # VGGish starts from 0.48 second

        feature_vector = []
        tf_label = []

        for chunk in range(vggish.shape[0]):
            start = chunk*half_sec

            vggish_means = np.mean(vggish[start:start+1, :], 0) # I removed the smooth to get only one window
            vggish_stddevs = np.std(vggish[start:start+1, :], 0)

            feature_vector.append(vggish[chunk, :])

            # Adjust labels to our classes
            if len([x for x in activation[start:start+feature_length] if x > 0]) >= half_sec: # 50%
                tf_label.append('present')
            else:
                tf_label.append('abscent')

        # Get labels index
        tf_label_ind = [labels.index(lbl) for lbl in tf_label]
        ### print("file label size: {:d}".format(len(tf_label_ind)))

        # Store the feature vector and corresponding label in integer format
        for idx in range(len(feature_vector)):
            load_features.append(feature_vector[idx])
            load_labels.append(tf_label_ind[idx]) 

    return (load_features, load_labels)

# Calculate or load features from train set
'''Serialize a transformation object or pipeline.

    Parameters
    ----------
    transform : BaseTransform or Pipeline
        The transformation object to be serialized

    kwargs
        Additional keyword arguments to `jsonpickle.encode()`

    Returns
    -------
    json_str : str
        A JSON encoding of the transformation

    See Also
    --------
    deserialize

    Examples
    --------
    >>> D = muda.deformers.TimeStretch(rate=1.5)
    >>> muda.serialize(D)
    '{"params": {"rate": 1.5},
      "__class__": {"py/type": "muda.deformers.time.TimeStretch"}}'
    '''

# Train the scaler
def define_scaler(train_features, filename):
    print (" == Scaler phase ==")
    # Create a scale object
    scaler = sklearn.preprocessing.StandardScaler()

    # Learn the parameters from the training data only
    scaler.fit(train_features)
    
    # save the scaler to disk
    joblib.dump(scaler, filename)

    return scaler
    
# Train the models 
def train_model_SVM(train_features_scaled, train_labels, split):
    print (" == Training phase ==")
    # Use scikit-learn to train a model with the training features we've extracted
    models = []
    # Lets use a SVC with folowing C parameters: 
    params = [10, 1, 0.1, 0.01]

    for c in params:
        clf = sklearn.svm.SVC(C=c)

        # Fit (=train) the model
        clf.fit(train_features_scaled, train_labels)

        # save the model to disk
        filename = 'models/finalized_model_'+str(split)+'_SVM_'+str(c)+'_VGGish.sav'
        print (filename)
        joblib.dump(clf, filename)

        models.append([clf, c])

    return models
        
        

""" This part could be done separately to be independent of input """
# Evaluate the models with validation set
def predict_model(clf, test_features_scaled, test_labels):
    print (" == Prediction phase ==")
    # Now lets predict the labels of the test data!
    predictions = clf.predict(test_features_scaled)
    
    return predictions
       
    
def evaluate_models(models, test_features_scaled, test_labels):
    print (" == Evaluation phase ==")
    results = []
    conf_mtx = []
    
    for clf, c in models:
        predictions = predict_model(clf, test_features_scaled, test_labels)
        
        cm = confusion_matrix(test_labels, predictions)
        
        conf_mtx.append([c, cm])
        
        # We can use sklearn to compute the accuracy score
        acc = sklearn.metrics.accuracy_score(test_labels, predictions)
        print ("Trained model with C-value", c,"has accuracy", acc)
        results.append([c, acc])
        
    return (results, conf_mtx)
    

# Save results to plot a graph

# Read file with splits
def read_split_file():
    # Store all results
    res_final = []
    cm_final = []
    
    # Create a list of all musics  
    with open('split_voiced_medleydb.json') as json_file:
        
        data = json.load(json_file)
        
        for split in range(len(data)):
            
            train_files = []
            validation_files = []
            test_files = []
    
            print ("\n >>> Running for split number >>> ", split)
        
            for music in data[split]["train"]:
                train_files.append(music)
            for music in data[split]["validation"]:
                validation_files.append(music)
            for music in data[split]["test"]:
                test_files.append(music)
                
                
            #**** It will happen within the for ****       
            print ("\n ** Features of train files ** ")
            train_features, train_labels = load_features(train_files, "VGGish_PCA")
            print ("\n ** Features of validation files ** ")
            validation_features, validation_labels = load_features(validation_files, "VGGish_PCA")
            print ("\n ** Features of test files ** ")
            test_features, test_labels = load_features(test_files, "VGGish_PCA")


            filename = 'scaler_'+str(split)+'_VGGish.sav'
            scaler = define_scaler(train_features, filename)

            # Apply the learned parameters to the training, validation and test sets:
            train_features_scaled = scaler.transform(train_features)

            validation_features_scaled = scaler.transform(validation_features)

            test_features_scaled = scaler.transform(test_features)
            #****                               ****

            models = train_model_SVM(train_features_scaled, train_labels, split)

            res_validation, cm_validation = evaluate_models(models, validation_features_scaled, validation_labels)
            print ("Validation accuracy", res_validation)
            res_test, cm_test = evaluate_models(models, test_features_scaled, test_labels) 
            print ("Test accuracy", res_test)
            res_final.append([split, res_validation, res_test])
            cm_final.append([split, cm_validation, cm_test])
        
        with open("cls_results.csv", 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerows(res_final)
        print ("Saved results.")
        
        with open("VGGish_SVM.csv", 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerows(cm_final)
        print ("Saved results.")
        
        
        """
        res_final = [
                    [split, [res_validation = 
                                            [[par, acc],...,[par, acc]]
                            ], 
                            [res_test = 
                                            [[par, acc],...,[par, acc]]
                            ]
                    ]
        
        """
        
        """
        cm_final = [
                    [split, [cm_validation = 
                                            [[par, cm],...,[par, cm]]
                            ], 
                            [cm_test = 
                                            [[par, cm],...,[par, cm]]
                            ]
                    ]
        
        """

if __name__ == "__main__":
    
    read_split_file()