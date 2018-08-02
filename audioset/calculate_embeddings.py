### This file in in ./models/research/audioset/


from __future__ import print_function

import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import librosa
import sys
import pandas as pd
import csv

print('\nTesting your install of VGGish\n')

# Paths to downloaded VGGish files.
checkpoint_path = './vggish_model.ckpt'
pca_params_path = './vggish_pca_params.npz'

# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

## Generate a 1 kHz sine wave at 44.1 kHz (we use a high sampling rate
## to test resampling to 16 kHz during feature extraction).
#num_secs = 3
#freq = 1000
#sr = 44100
#t = np.linspace(0, num_secs, int(num_secs * sr))
#x = np.sin(2 * np.pi * freq * t)

file_path = sys.argv[1]
x, sr = librosa.load(file_path, sr=None)
num_secs = librosa.get_duration(x, sr)
print (len(x)*sr)
print (librosa.get_duration(x,sr))

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(x, sr)
print('Log Mel Spectrogram example: ', input_batch[0])
#np.testing.assert_equal(
#    input_batch.shape,
#    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

# Define VGGish, load the checkpoint, and run the batch through the model to
# produce embeddings.
config = tf.ConfigProto(
                device_count = {'GPU': 0}
                    )
with tf.Graph().as_default(), tf.Session(config=config) as sess:
  vggish_slim.define_vggish_slim()
  vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

  features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
  embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
  [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
  print('VGGish embedding: ', embedding_batch[0])

  # Postprocess the results to produce whitened quantized embeddings.
  pproc = vggish_postprocess.Postprocessor(pca_params_path)
  postprocessed_batch = pproc.postprocess(embedding_batch)
  print('Postprocessed VGGish embedding: ', postprocessed_batch[0])

  results_path = file_path[:-7] + "VGGish_PCA.csv"
  with open(results_path, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(postprocessed_batch.shape[0]):
      spamwriter.writerow(postprocessed_batch[i])
  print ("Saved VGGish embeddings.")

  print ('Embedding shape: ', embedding_batch.shape)


