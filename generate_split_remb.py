import pandas as pd
import glob, os

split_remb = {}

# Get a list of all the training audio files (must be .WAV files)
dataset_folder = '/home/shayenne/Documents/rembDB_labeledExamples/'
dataset_artists = glob.glob(os.path.join(dataset_folder, '*'))

# Specify where the audio files for training and testing reside (80% train / 20% test)
train_folders = [os.path.basename(f) for f in dataset_artists[:17]]
test_folders = [os.path.basename(f) for f in dataset_artists[17:]]

split_remb['train'] = train_folders
split_remb['test'] = test_folders

import json

with open('split_remb.json', 'w') as f:
    json.dump(split_remb, f)