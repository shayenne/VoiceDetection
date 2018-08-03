import librosa

"""
"    Features calculated without a library
"""
def calculate_plp():
    pass

def calculate_lpc():
    pass

def calculate_vv():
    pass

def calculate_fluctogram():
    pass


"""
"   Features from librosa
"""

def calculate_hpss(D):
    return  librosa.decompose.hpss(D)

def calculate_melspectrogram(y, sr):
    return librosa.feature.melspectrogram(y, sr)

def calculate_mfcc(y, sr):
    return librosa.feature.melspectrogram(y, sr, n_mels=128, fmax=8000)

def calculate_rmse(y):
    return librosa.feature.rmse(y)

def calculate_spec_centroid(y, sr):
    return librosa.feature.spectral_centroid(y=y, sr=sr)

def calculate_spec_bandwidth(y, sr):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)

def calculate_spec_contrast(y, sr):
    return librosa.feature.spectral_contrast(y=y, sr=sr)

def calculate_spec_flatness(y, sr):
    return librosa.feature.spectral_flatness(y=y, sr=sr)

def calculate_spec_rolloff(y, sr, window, hop):
    return librosa.feature.spectral_rolloff(y, sr, hop_length=hop)

def calculate_zcr(y, window, hop):
    return librosa.feature.zero_crossing_rate(y, frame_length=window, hop_length=hop)
    

def calculate_delta(ftr):
    return librosa.feature.delta(ftr)


"""
"   Features calculated from other sources
"""

def calculate_vggish():
    pass


"""
"   Calculate all features 
"""
def calculate_all_features():
    # put all features together
    pass


def main():
    # Audio files in a list (path) 
    files = []

    # Open files directors and save the calculated features
    for f in files:
        # load audio

        # calculate features
        calculate_all_features(f)

        # save on the same path


def test():
    window = 4410
    hop = 2205

    y, sr = librosa.load("../LizNelson_Rainfall_MIX.wav", sr=None)
    print ("Loaded audio")
    feature = calculate_zcr(y, window, hop)
    print (feature)
