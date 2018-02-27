import paths                        # Environment paths
import os                           # Manipulate paths
import numpy as np                  # Numerical processing
import librosa                      # Audio processing
import matplotlib.pyplot as plt     # Plot graphs
import csv                          # Manipulate csv files
import librosa.display
import pandas as pd

import paths


"""
Calculate features
"""
def calculate_features():

    # All files selected from MedleyDB
    audio_path = os.path.join(os.environ['AUDIO_PATH'])

    test = 0
    for filename in os.listdir(audio_path):
        if filename.endswith(".wav"):
            music = filename
            music = music[:-4]
        else:
            continue

        print ("--- Preprocessing...")
        print (filename)

        # Load audio file
        y, orig_sr = librosa.load(audio_path+filename, mono=True) 

        # Resample
        #target_sr = 8000
        #y_res = librosa.resample(y, orig_sr, target_sr)

        print ("> Audio signal loaded...")
        hl = int(0.0020*orig_sr) # 20 milisseconds
        wl = int(0.0040*orig_sr) # 40 milisseconds
        
        # STFT analysis (Parameters based on article [1])
        S = librosa.stft(y, n_fft=1024, hop_length=hl,
                         win_length=wl, window='hann')
        D = librosa.amplitude_to_db(S, ref=np.max)

        # Remove frequencies above 2kHz
        #D_cut = D[:256, :]

        

        print("Writing features file...")

    
        # Write features file
        feature_path = os.path.join(os.environ['FEATURE_PATH'],
                                    music+'_features.csv')
    
        with open(feature_path, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            col = D.shape[1]
            for i in range(col):
                spamwriter.writerow(D[:,i])
        print("DONE")

        test +=1

        if test > 0:
            break

"""
Load all files
"""

# Input Data
def load_data(feature_path=None, label_path=None):
    """ Receives features and labels path and return it on a list [X,y],
        where 'X' has the feature matrix and 'y' has the vector label.
    """

    if feature_path is None:
        feature_path = os.path.join(os.environ['FEATURE_PATH'])
    if label_path is None:
        label_path = os.path.join(os.environ['LABEL_PATH'])

    list_feat = []
    list_lbl  = []
    i = 0
    for filename in os.listdir(feature_path):
        if filename.endswith(".csv"):
            music = filename
            music = music[:-12] # Remove 'features.csv'
        else:
            continue
        # Read STFT from csv files
        print("   Load", feature_path + music + "features.csv")
        d1 = pd.read_csv(feature_path + music + "features.csv",
                         index_col=None, header=None)
        list_feat.append(d1)
        print("   Load", label_path + music + "labels.csv")
        d2 = pd.read_csv(label_path + music + "labels.csv",
                         index_col=None, header=None)
        list_lbl.append(d2)
        # Fitting each music
        print(d1.shape, d2.shape)
        i += 1

        if i == 1:
            break

    # Grouping data
    X = pd.concat(list_feat)
    y = pd.concat(list_lbl)

    # Return data
    return [X, np.ravel(y)]


"""
    Classifiers
"""

# All true
def all_true_predict(X):
    return np.ones(len(X))


# Energy threshold
threshold = None
def energy_threshold_fit(X_train, y_train):
    global threshold    
    threshold = 0.8
    pass

def energy_threshold_predict(X):
    global threshold
    print (threshold)
    return [1 if x is True else 0 for x in (X >= threshold)]

def find_threshold(ref_values, energy):
    """ Find the best threshold for the data - IS IT RIGHT? """
    last_acc = 0 # Start accuracy
    steps = 1000
    
    R, P, Acc, F1, FA = [],[],[],[],[]
    
    for threshold in range(steps):
        # Apply threshold
        est_values = np.where(energy>(threshold/steps), 1, 0)
        
        # Calculate measures
        TP = (ref_values*est_values).sum()
        FP = ((ref_values == 0)*est_values).sum()
        FN = (ref_values*(est_values == 0)).sum()
        TN = ((ref_values == 0)*(est_values == 0)).sum()
        
        R.append(TP/(TP+FN))
        P.append(TP/(TP+FP))
        
        Acc.append((TP+TN)/(TP+TN+FP+FN))
        F1.append(2*P[-1]*R[-1]/(P[-1]+R[-1]))
        FA.append(FP/(FP+TN))
        
        # Verify best values
        if last_acc <= Acc[-1]:
            last_acc = Acc[-1]
            best = (threshold/steps)
            output = est_values

    # Plot measures     
    plt.plot(P, label='P')
    plt.plot(R, label='R')
    plt.plot(F1, label='F1')
    plt.plot(FA, label='FA')
    plt.plot(Acc, label='Acc')
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
        
    return best, output

def evaluate_results(ref_values, est_values):
    """ Receives the ref_values and est_values """
    # Calculate measures
    TP = (ref_values*est_values).sum()
    FP = ((ref_values == 0)*est_values).sum()
    FN = (ref_values*(est_values == 0)).sum()
    TN = ((ref_values == 0)*(est_values == 0)).sum()
            
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F = 2*P*R/(P+R)
    FA = FP/(TN+FP)
    VA = TP/(TP+FN)
    OA = (TP + TN) / (len(lbl))
    print ("Precision:    {}".format(P))
    print ("Recall:       {}".format(R))
    print ("F-measure:    {}".format(F))
    print ("Voicing False Alarm Rate:  {}".format(FA))  
    print ("Voicing Recall Rate:       {}".format(VA))
    print ("Overall Accuracy:          {}".format(OA))
