import paths                        # Environment paths
import os                           # Manipulate paths
import numpy as np                  # Numerical processing
import librosa                      # Audio processing
import matplotlib.pyplot as plt     # Plot graphs
import csv                          # Manipulate csv files

# All files selected from MedleyDB
audio_path = os.path.join(os.environ['AUDIO_PATH'])

for filename in os.listdir(audio_path):
    if filename.endswith(".wav"):
        music = filename
        music = music[:-8]
    else:
        continue

    print ("--- Preprocessing...")
    print (filename)

    # Load audio file
    y, orig_sr = librosa.load(audio_path+filename, mono=True) 

    # Resample
    target_sr = 8000
    y_res = librosa.resample(y, orig_sr, target_sr)

    print ("> Audio signal loaded...")

    # STFT analysis (Parameters based on article [1])
    S = librosa.stft(y_res, n_fft=1024, hop_length=64,
                     win_length=1024, window='hann')
    D = librosa.amplitude_to_db(S, ref=np.max)

    # Remove frequencies above 2kHz
    D_cut = D[:256, :]

    print("Writing features file...")

    
    # Write features file
    feature_path = os.path.join(os.environ['FEATURE_PATH'],
                                music+'_features.csv')
    
    with open(feature_path, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        col = D_cut.shape[1]
        for i in range(col):
            spamwriter.writerow(D_cut[:,i])
    print("DONE")


    """
    "   Format labels for input data
    "
    """

    # Format label
    lbl = []


    """ Labels """
    # Annotation 1 - Save with double frequency
    label_path = os.path.join(os.environ["LABEL_PATH"],
                              music+'_labels.csv')
    annot_path = os.path.join(os.environ["ANNOT_PATH"],
                              music+"_MELODY1.csv")


    # Read csv file
    melody = []
    timestamps = []
    
    with open(annot_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            timestamps.append(float(row[0]))
            melody.append(float(row[1]))

            
    # Append element to make size equal of spectrogram
    melody.append(0)
    timestamps.append(0)

    
    # None to values less equals 0
    melody_pos = melody[:].copy()
    for i in range(len(melody_pos)):
        if melody_pos[i] <= 0:
            melody_pos[i] = None


    # Convert annotation data to 16kHz [VERIFY IF IT WORKS!]
    Horig  = 256 # From MedleyDB
    SRorig = orig_sr
    
    Hnew  = 64 # From article [1] * EDIT
    SRnew = target_sr
    
    size = D_cut.shape[1]
    
    j = np.arange(size) * (SRorig/Horig * Hnew/SRnew)


    # Resample to 8kHz
    # None to values less equals 0
    melody_res = np.zeros(len(j))
    tmstamps = np.zeros(len(j))
    for i in range(len(j)):
        # Get label more near from this frame resampled
        melody_res[i] = melody[int(j[i])]   
        tmstamps[i] = timestamps[int(j[i])]
        if melody_res[i] <= 0:
            melody_res[i] = 0
            lbl.append(0)
        else:
            lbl.append(1)

    print (len(melody_res), len(tmstamps))


    T = np.arange(193) # Adjust for the real purpuose!!!
    # Value 4 is log2(1108.73) - log2(69.29)
    T1 = np.linspace(0,4,193)


    
    """ ALERT: FUNCTION HERE!!! """
    # Define what value will be get from original annotation
    def find_nearest(array,value):
        i = (np.abs(array-value)).argmin()
        return i
    """ ----------------------- """


    print("Writing labels file...")
    # Save labels file
    with open(label_path, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(melody_res)):
            if melody_res[i] == 0:
                spamwriter.writerow("0")
            else:
                label = find_nearest(T1,
                                     np.log2(melody_res[i])-np.log2(69.29))
                spamwriter.writerow([label]) 
    print("DONE")
