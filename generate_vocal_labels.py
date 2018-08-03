import pandas as pd
import numpy as np
import json
import librosa as lr
import os


# Path for source activations
SOURCEID_PATH = "MedleyDB/Annotations/Instrument_Activations/SOURCEID/"
AUDIO_PATH = "MedleyDB/Audio/"

# Process files to create label and save on a given path
def save_labels(files, dir_path):
        
    # Process musics
    for music in files:
        source_activation = pd.read_csv(SOURCEID_PATH + music +\
                                        "_SOURCEID.lab", index_col=None)

        y, sr = lr.load(AUDIO_PATH + music + "/" + music + "_MIX.wav", sr=None)
        duration = lr.get_duration(y, sr)

        # Get music duration in miliseconds
        label_vector = np.zeros(int(duration*100)) 

        for idx, source in source_activation.iterrows():
            
            if source.instrument_label in ['female singer', 'male singer', \
                                           'vocalists', 'choir']:
                start, end = source.start_time, source.end_time
                label_vector[int(start*100):int(end*100)] = 1
                
        df = pd.DataFrame(label_vector.astype('int').T,columns=None,index=None)

        # Save vocal labels and copy the audio file to the right place
        df.to_csv(dir_path + music+"_vocal.csv", index=False, header=False)
        os.system('cp ' + AUDIO_PATH + music + '/' + music + '_MIX.wav' \
                  + ' ' + dir_path)

        print ("> Vocal labels for", dir_path,"completed.")


if __name__ == "__main__":
    # Create a list of all musics
    train_files = []
    validation_files = []
    test_files = []
    with open('split_vocal_medleydb.json') as json_file:  
        data = json.load(json_file)
        
        for music in data["train"]:
            train_files.append(music)
        for music in data["validation"]:
            validation_files.append(music)
        for music in data["test"]:
            test_files.append(music)

    # Saving vocal labels 
    save_labels(train_files, "exp1/train/")
    save_labels(validation_files, "exp1/validation/")
    save_labels(test_files, "exp1/test/")
