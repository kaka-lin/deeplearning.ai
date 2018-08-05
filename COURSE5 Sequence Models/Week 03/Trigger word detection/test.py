import os
import sys
import librosa

from kaka_utils import *
from td_utils import *

def test():
    file_path = "audio_examples/example_train.wav"
    x = graph_spectrogram(file_path)
    print(x.shape)
    
    activates, negatives, backgrounds = load_raw_audio()

    # Because speech data is hard to acquire and label.
    # We will synthesize our data
    # Insert the word's short audio clip onto the background.
    np.random.seed(5)
    audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
    audio_clip.export("insert_test.wav", format="wav")
    print("Segment Time:", segment_time)
    play_wave("insert_test.wav")

    Ty = 1375
    arr1 = insert_ones(np.zeros((1, Ty)), 9700)
    plt.figure()
    plt.plot(insert_ones(arr1, 4251)[0,:])
    print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])
    plt.show()

if __name__ == '__main__':
    test()