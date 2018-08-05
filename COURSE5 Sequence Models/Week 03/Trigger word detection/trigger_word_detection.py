import os
import sys

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

from kaka_utils import *

def build_model(input_shape):
    model = Sequential()

    # Step 1: CONV layer: 5511 -> 1375
    model.add(Conv1D(196, 15, strides=4, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))

    # Step 2: First GRU Layer
    model.add(GRU(units=128, return_sequences=True)) # GRU (use 128 units and return the sequences)
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    
    # Step 3: Second GRU Layer (≈4 lines)
    model.add(GRU(units=128, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    
    # Step 4: Time-distributed dense layer (≈1 line)
    model.add(TimeDistributed(Dense(units=1, activation='sigmoid'))) # time distributed  (sigmoid)

    return model

def train_model(x, y, batch_size, epochs, verbose=1, retrain=False, save=False):
    """ kaka_model
    
    batch_size:5, epochs:500
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    
    batch_size:5, epochs:100
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    
    batch_size:5, epochs:200
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
     
    loss: 0.12, acc: 0.96
    
    """
    if retrain:
        model = load_model('./models/kaka_model.h5')
    else:
        model = build_model(input_shape=(5511, 101))
    
    model.summary()
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
    
    # train and save model
    model.fit(x=x, y=y, 
              batch_size=batch_size, 
              epochs=epochs,
              verbose=verbose)

    if save:
        model_name = 'kaka_model_1'
        save_model(model, model_name=model_name)
        print("Save model to models/{}".format(model_name))


if __name__ == '__main__':
    # Load preprocessed training examples
    x = np.load("./XY_train/X.npy") # (26, 5511, 101)
    y = np.load("./XY_train/Y.npy") # (26, 1375, 1)

    # Load preprocessed dev set examples 
    x_dev = np.load("./XY_dev/X_dev.npy") # (26, 5511, 101)
    y_dev = np.load("./XY_dev/Y_dev.npy") # (26, 1375, 1)

    # train model
    batch_size = 5
    epochs = 100
    #train_model(x=x, y=y, batch_size=batch_size, epochs=epochs)

    # test model
    for model_name in os.listdir('./models'):
        if model_name.endswith('h5'):
            model = load_model('./models/' + model_name)
            print('='*10, model_name, '='*10)
            loss, acc = model.evaluate(x_dev, y_dev)
            print("Dev set accuracy = {}, loss = {}".format(acc, loss))

            '''
            # Making Predictions
            filename = "./raw_data/dev/1.wav"
            prediction = detect_triggerword(model, filename)
            chime_on_activate(filename, prediction, 0.5)
            play_wave("./chime_output.wav")
            '''
    
            # Try your own example
            filename = "audio_examples/my_audio.wav"
            preprocess_audio(filename)
            play_wave(filename)
            
            chime_threshold = 0.5
            prediction = detect_triggerword(model, filename)
            chime_on_activate(filename, prediction, chime_threshold)
            play_wave("./chime_output.wav")
            