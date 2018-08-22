import pandas as pd
import numpy as np
import json
import librosa
import os

from paths import *

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



# Read file with splits
def read_split_file():
    # Create a list of all musics  
    with open('split_voiced_medleydb.json') as json_file:
        
        data = json.load(json_file)
        
        for idx in range(len(data)):
            
            train_files = []
            validation_files = []
            test_files = []
    
            print ("\n >>> Running for split number >>> ", idx)
        
            for music in data[idx]["train"]:
                train_files.append(music)
            for music in data[idx]["validation"]:
                validation_files.append(music)
            for music in data[idx]["test"]:
                test_files.append(music)
                
            print ("\n ** Features of train files ** ")
            load_features(train_files, "VGGish_PCA")
            print ("\n ** Features of validation files ** ")
            load_features(validation_files, "VGGish_PCA")
            print ("\n ** Features of test files ** ")
            load_features(test_files, "VGGish_PCA")

            
            
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


# Train the models 


""" This part could be done separately to be independent of input """
# Evaluate the models with validation set

# Save results to plot a graph




if __name__ == "__main__":
    
    read_split_file()