import os
import errno

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from pydub import AudioSegment
from scipy.io import wavfile
import wave

SAMPLE_RATE = 44100
DURATION = 10.0
HOP_LENGTH = 512
# MFCC -> (n_mfcc, t)
# t = sample_rate * time / hop_length
MAX_LENGTH = int((SAMPLE_RATE * DURATION // HOP_LENGTH) + 1)

#########################################################################################
"""
To implement the training set synthesis process, 
you will use the following helper functions. 
All of these function will use a 1ms discretization interval, 
so the 10sec of audio is alwsys discretized into 10,000 steps.

1. get_random_time_segment(segment_ms):
    gets a random time segment in our background audio
2. is_overlapping(segment_time, existing_segments):
    checks if a time segment overlaps with existing segments
3. insert_audio_clip(background, audio_clip, existing_segments):
    inserts an audio segment at a random time in our background audio using get_random_time_segment and is_overlapping
4. insert_ones(y, segment_end_ms):
    inserts 1's into our label vector y after the word "activate"

"""

def get_random_time_segment(segment_ms):
    """ 
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    @param segment_ms:    the duration of the audio clip in ms
    
    @return segment_time: a tuple of (segment_start, segment_end) in ms
    
    """

    # Make sure segment doesn't run past the 10sec background 
    segment_start = np.random.randint(low=0, high=10000-segment_ms)
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

def is_overlapping(segment_time, existing_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    @param segment_time:      a tuple of (segment_start, segment_end) for the new segment
    @param existing_segments: a list of tuples of (segment_start, segment_end) 
                              for the existing segments
    
    @return overlap:          True if the time segment overlaps 
                              with any of the existing segments, 
                              False otherwise

    """

    segment_start, segment_end = segment_time

    overlap = False
    for existing_start, existing_end in existing_segments:
        if segment_start <= existing_end and segment_end >= existing_start:
            overlap = True
    
    return overlap

def insert_audio_clip(background, audio_clip, existing_segments):
    """
    Insert a new audio segment over the background noise at a random time step, 
    ensuring that the audio segment does not overlap with existing segments.
    
    @param: background:       a 10 second background audio recording.  
    @param audio_clip:        the audio clip to be inserted/overlaid. 
    @param existing_segments: times where audio segments have already been placed
    
    @return new_background:   the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, existing_segments):
        segment_time = get_random_time_segment(segment_ms)
    
    existing_segments.append(segment_time)

    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time

def insert_ones(y, segment_end_ms, Ty=1375, duration=10000.0):
    """ Update the label vector y.

    The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. 
    By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
   
    @param y:              numpy array of shape (1, Ty), 
                           the labels of the training example
    @param segment_end_ms: the end time of the segment in ms
    
    @return y: updated labels
    """

    # ex: 9700: 10000 = x : 1375
    #     => x = (9700 * 1375) / 10000.0
    segment_end_y = int(segment_end_ms * Ty / duration)
    for i in range(segment_end_y+1, segment_end_y+51):
        if i < y.shape[1]:
            y[0, i] = 1
    
    return y

def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.
    
    @param background: a 10 second background audio recording
    @param activates:  a list of audio segments of the word "activate"
    @param negatives:  a list of audio segments of random words that are not "activate"
    
    @return x:         the spectrogram of the training example
    @return y:         the label at each time step of the spectrogram

    """

    # Set the random seed
    np.random.seed(18)

    # Make background quieter
    background = background - 20

    # Step 1: Initialize y (label vector) of zeros
    y = np.zeros((1, 1375))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    existing_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, existing_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end)
    
    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, existing_segments)
    
    # Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)

    # Export new training example 
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")
    
    return x, y
    
#########################################################################################
# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds

#########################################################################################
# MFCC
def wav2mfcc(file_path, sr=None, offset=0.0, duration=None, n_mfcc=13, max_length=MAX_LENGTH):
    data, sr = librosa.load(file_path, mono=True, sr=sr, offset=offset, duration=duration)
    mfcc = librosa.feature.mfcc(data, sr=sr, n_mfcc=n_mfcc)
    #S = librosa.feature.melspectrogram(data, sr=sr, n_mels=128, fmax=8000)
    
    if (max_length > mfcc.shape[1]):
        #print(max_length, mfcc.shape[1])
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    
    '''
    # plot
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.waveplot(data, sr=sr)
    plt.subplot(2,1,2)
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    #plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    '''
    '''
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             y_axis='mel',
                             x_axis='time',
                             fmax=8000)
    '''
    
    return mfcc

def play_wave(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]

    p = pyaudio.PyAudio()

    # Convert audio to audio stream
    stream = p.open(format =  p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)

    # define chunk
    chunk = 1024

    # read audio data and play stream
    data = f.readframes(chunk)
    while data != b'':
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()
    
    #close PyAudio
    p.terminate()

#########################################################################################
def save_model(model, model_name):
    file_path = 'models/{}.h5'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    model.save(file_path)

#########################################################################################
def detect_triggerword(model, filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    '''
    Once you've estimated the probability of having detected 
    the word "activate" at each output step, 
    you can trigger a "chiming" sound to play 
    when the probability is above a certain threshold. 
    Further,  y⟨t⟩ might be near 1 for many values in a row after "activate" is said, 
    yet we want to chime only once. 
    So we will insert a chime sound at most once every 75 output steps. 
    This will help prevent us from inserting two chimes for a single instance of "activate". 
    (This plays a role similar to non-max suppression from computer vision.)
    '''
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')

# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')