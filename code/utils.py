############################################
# LIBRARIES
############################################

from scipy.io import wavfile
import tensorflow as tf
import numpy as np
import os
import scipy.signal
import librosa
import random
import time
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)


############################################
# DATA GENERATORS
############################################

class DataGeneratorSyntheticDataset(tf.keras.utils.Sequence):

    def __init__(self, dataframe, dir_farend_speech, dir_nearend_mic, dir_nearend_speech, batch_size=1, shuffle=False, mode='train'):

        # Direct inputs
        self.dataframe = dataframe
        self.dir_farend_speech = dir_farend_speech
        self.dir_nearend_mic = dir_nearend_mic
        self.dir_nearend_speech = dir_nearend_speech
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode

        # Secondary Inputs
        self.files = list(self.dataframe['signal_name'])
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        ''' Returns the number of batches per epoch '''
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        ''' Returns a batch of data (the batches are indexed) '''
        # Take the id's of the batch number "index"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Batch initialization
        X, Y = [], []

        # For each index,the sample and the label is taken. Then the batch is appended
        for idx in indexes:
            # Image and idx index tag is get
            if self.mode == 'test':
                x, y = self.get_sample_signal(idx)
            else:
                x, y = self.get_sample_spectrogram(idx)
            # This image to the batch is added
            X.append(x)
            Y.append(y)
        # The created batch is returned
        return np.concatenate(X, 0), np.concatenate(Y, 0)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes) # Shuffles the data

    def get_sample_spectrogram(self, idx):

        # Sampling frecuency
        fs = 16000

        # Buffer length in ms converted to samples
        n0 = int(120*(1e-3)* fs)
        n_prev = int(115*(1e-3)* fs)
        n_post = int(40*(1e-3)* fs)

        # Load signals
        samplerate_farend_speech, farend_speech = readwav(self.dir_farend_speech + '/' + 'farend_speech_fileid_' + self.files[idx])
        samplerate_nearend_mic, nearend_mic = readwav(self.dir_nearend_mic + '/' + 'nearend_mic_fileid_' + self.files[idx])
        samplerate_nearend_speech, nearend_speech = readwav(self.dir_nearend_speech + '/' + 'nearend_speech_fileid_' + self.files[idx])

        # Loop to load images
        X = []
        Y = []
        n_buffers = int((farend_speech.shape[0]-n0)/(n_prev+n_post))

        for iBuffer in range(0, n_buffers):

            # Get buffers
            trama_farend_speech = farend_speech[n0+iBuffer*(n_prev+n_post):n0+(iBuffer+1)*(n_prev+n_post)]
            trama_nearend_mic = nearend_mic[n0 + iBuffer * (n_prev + n_post):n0 + (iBuffer + 1) * (n_prev + n_post)]
            trama_nearend_speech = nearend_speech[n0 + iBuffer * (n_prev + n_post):n0 + (iBuffer + 1) * (n_prev + n_post)]

            # Spectrogram
            stft_farend_speech, _ = spectrogram(trama_farend_speech, samplerate_farend_speech)
            stft_nearend_mic, _ = spectrogram(trama_nearend_mic, samplerate_nearend_mic)
            stft_nearend_speech, _ = spectrogram(trama_nearend_speech, samplerate_nearend_speech)

            # Reshape to create volume
            stft_farend_speech = np.reshape(stft_farend_speech, (stft_farend_speech.shape[0], stft_farend_speech.shape[1], 1))
            stft_nearend_mic = np.reshape(stft_nearend_mic, (stft_nearend_mic.shape[0], stft_nearend_mic.shape[1], 1))
            stft_nearend_speech = np.reshape(stft_nearend_speech, (stft_nearend_speech.shape[0], stft_nearend_speech.shape[1], 1))

            # Concatenate
            x = np.concatenate([stft_farend_speech, stft_nearend_mic], axis=2)
            y = stft_nearend_speech

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)

        # Select randomly 30 samples
        idx = random.sample(range(0, 61), 32)
        X = X[idx, :, :, :]
        Y = Y[idx, :, :, :]

        return X, Y

    def get_sample_signal(self, idx):

        samplerate_farend_speech, farend_speech = readwav(self.dir_farend_speech + '/' + 'farend_speech_fileid_' + self.files[idx])
        samplerate_nearend_mic, nearend_mic = readwav(self.dir_nearend_mic + '/' + 'nearend_mic_fileid_' + self.files[idx])
        samplerate_nearend_speech, nearend_speech = readwav(self.dir_nearend_speech + '/' + 'nearend_speech_fileid_' + self.files[idx])

        farend_speech = np.reshape(farend_speech, (farend_speech.shape[0], 1))
        nearend_mic = np.reshape(nearend_mic, (nearend_mic.shape[0], 1))
        nearend_speech = np.reshape(nearend_speech, (nearend_speech.shape[0], 1))

        X = np.reshape(np.concatenate([farend_speech, nearend_mic], axis=1), (1, farend_speech.shape[0], 2))
        Y = np.reshape(nearend_speech, (1, nearend_speech.shape[0]))

        return X, Y


class DataGeneratorTestDataset(tf.keras.utils.Sequence):

    def __init__(self, dir_data, dataframe, batch_size=1, shuffle=False, mode='train'):

        # Direct inputs
        self.dir_data = dir_data
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode

        # Secondary Inputs
        self.files = list(self.dataframe['signal_name'])
        self.indexes = np.arange(len(self.files))

    def __len__(self):
        ''' Returns the number of batches per epoch '''
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        ''' Returns a batch of data (the batches are indexed) '''
        # Take the id's of the batch number "index"
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Batch initialization
        X, Y = [], []

        # For each index,the sample and the label is taken. Then the batch is appended
        for idx in indexes:
            # Image and idx index tag is get
            if self.mode == 'test':
                x, y = self.get_sample_signal(idx)
            else:
                x, y = self.get_sample_spectrogram(idx)
            # This image to the batch is added
            X.append(x)
            Y.append(y)
        if self.mode == 'test':
            X = np.concatenate(X, 0)
            Y = np.array(Y)
        else:
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)

        # The created batch is returned
        return X, Y

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes) # Shuffles the data

    def get_sample_spectrogram(self, idx):

        # Sampling frecuency
        fs = 16000

        # Buffer length in ms converted to samples
        n0 = int(120*(1e-3)* fs)
        n_prev = int(115*(1e-3)* fs)
        n_post = int(40*(1e-3)* fs)

        # Load signals
        if 'clean' in self.files[idx]:
            samplerate_nearend_mic, nearend_mic = readwav(self.files[idx] + '_mic_c.wav')
            yy = 1
        elif 'noisy' in self.files[idx]:
            samplerate_nearend_mic, nearend_mic = readwav(self.files[idx] + '_mic.wav')
            yy = 0

        # Loop to load images
        X = []
        Y = []
        n_buffers = int((nearend_mic.shape[0]-n0)/(n_prev+n_post))

        for iBuffer in range(0, n_buffers):

            # Get buffers
            trama_nearend_mic = nearend_mic[n0 + iBuffer * (n_prev + n_post):n0 + (iBuffer + 1) * (n_prev + n_post)]

            # Spectrogram
            stft_nearend_mic, _ = spectrogram(trama_nearend_mic, samplerate_nearend_mic)

            # Reshape to create volume
            stft_nearend_mic = np.reshape(stft_nearend_mic, (stft_nearend_mic.shape[0], stft_nearend_mic.shape[1], 1))

            # Concatenate
            x = stft_nearend_mic
            y = yy

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)

        # Select randomly 30 samples
        idx = random.sample(range(0, X.shape[0]), 32)
        X = X[idx, :, :]
        Y = Y[idx]

        return X, Y

    def get_sample_signal(self, idx):

        # Load signals
        if 'clean' in self.files[idx]:
            samplerate_nearend_mic, nearend_mic = readwav(self.files[idx] + '_mic_c.wav')
            yy = 1
        elif 'noisy' in self.files[idx]:
            samplerate_nearend_mic, nearend_mic = readwav(self.files[idx] + '_mic.wav')
            yy = 0

        X = np.reshape(nearend_mic, (1, nearend_mic.shape[0]))
        Y = yy

        return X, Y


############################################
# DEEP LEARNING
############################################

def mse_coef(y_true, y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))

    return loss


def rmse_coef(y_true, y_pred):

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)))

    return loss


def rmse_coef_slicing(y_true, y_pred):

    y_true = tf.slice(y_true, [0, 0, 20, 0], [32, 160, 12, 1])
    y_pred = tf.slice(y_pred, [0, 0, 20, 0], [32, 160, 12, 1])

    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)

    loss = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)))

    return loss


def u_net_2d(input_size=(161, 33), optimizer='nadam', learning_rate=1e-5, mode=0, number_filters_0=64, resize_factor_0=[1, 1], res_factor=[2, 1]):

    # Mode = 0 --> sequential
    # Mode = 1 --> sequential 2 conv layers
    # Mode = 2 --> residual 1 layer
    # Mode = 3 --> sequential 2 layers
    # Mode = 4 --> blocks multi resolution for feature extraction

    # Blocks definition

    def residual_block_1(input_layer, n_filters):

        x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
        x = tf.keras.layers.Add()([x, input_layer])

        return x

    def residual_block_2(input_layer, n_filters):

        x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(x)
        x2 = tf.keras.layers.Conv2D(n_filters, 3, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Add()([x, x2])

        return x

    def convolutional_block_1(input_layer, n_filters, kernel_size=3):

        x = tf.keras.layers.Conv2D(n_filters, kernel_size, activation='relu', padding='same')(input_layer)

        return x

    def convolutional_block_2(input_layer, n_filters, stride=3):

        x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(x)

        return x

    def multi_res_block(input_layer, n_filters):

        x1 = tf.keras.layers.Conv2D(n_filters//4, 3, activation='relu', padding='same')(input_layer)
        x2 = tf.keras.layers.Conv2D(n_filters//4, 3, activation='relu', padding='same')(x1)
        x3 = tf.keras.layers.Conv2D(n_filters//2, 3, activation='relu', padding='same')(x2)
        xMultiRes = tf.keras.layers.concatenate([x1, x2, x3])
        x4 = tf.keras.layers.Conv2D(n_filters, 1, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Add()([xMultiRes, x4])

        return x

    def encoding_block(input_layer, pooling_factor, number_filters_0, filters_factor, mode_convolution=1):

        x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(input_layer)  # pooling
        #x = input_layer
        x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor, kernel_size=3)  # dimensionality normalization
        # Feature extraction block
        if mode == 0:
            x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 1:
            x = convolutional_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 2:
            x = residual_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 3:
            x = residual_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 4:
            x = multi_res_block(x, n_filters=number_filters_0*filters_factor)

        return x

    def decoding_block(input_layer, skip_connection_layer, pooling_factor, number_filters_0, filters_factor, mode_convolution=1):

        # Deconvolution
        x = tf.keras.layers.UpSampling2D(size=(pooling_factor[0], pooling_factor[1]))(input_layer)
        #x = input_layer
        x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor, kernel_size=3)
        # Skip connection and number of filters normalization
        x = tf.keras.layers.concatenate([skip_connection_layer, x])
        x = convolutional_block_1(x, n_filters=number_filters_0 * filters_factor, kernel_size=3)
        # Feature extraction block
        if mode == 0:
            x = convolutional_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 1:
            x = convolutional_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 2:
            x = residual_block_1(x, n_filters=number_filters_0*filters_factor)
        elif mode == 3:
            x = residual_block_2(x, n_filters=number_filters_0*filters_factor)
        elif mode == 4:
            x = multi_res_block(x, n_filters=number_filters_0*filters_factor)

        return x

    # Architecture definition

    inputs = tf.keras.Input((input_size[0], input_size[1], 2))

    # ----- ENCODING -----

    # Block 1, 1024 --> 512, Filters_0 x 1
    encoding_1_out = encoding_block(inputs, resize_factor_0, number_filters_0, 1, mode_convolution=mode)
    # Block 2, 512 --> 256, Filters_0 x 2
    encoding_2_out = encoding_block(encoding_1_out, res_factor, number_filters_0, 2, mode_convolution=mode)
    # Block 3, 256 --> 128, Filters_0 x 4
    encoding_3_out = encoding_block(encoding_2_out, res_factor, number_filters_0, 4, mode_convolution=mode)
    # Block 4, 128 --> 64, Filters_0 x 8
    encoding_4_out = encoding_block(encoding_3_out, res_factor, number_filters_0, 8, mode_convolution=mode)

    # ----- DECODING -----

    # Block 2, 64 --> 128, Filters_0 x 4
    decoding_2_out = decoding_block(encoding_4_out, encoding_3_out, res_factor, number_filters_0, 4, mode_convolution=mode)
    # Block 3, 128 --> 256, Filters_0 x 2
    decoding_3_out = decoding_block(decoding_2_out, encoding_2_out, res_factor, number_filters_0, 2, mode_convolution=mode)
    # Block 4, 256 --> 512, Filters_0 x 1
    decoding_4_out = decoding_block(decoding_3_out, encoding_1_out, res_factor, number_filters_0, 1, mode_convolution=mode)

    # ----- OUTPUT -------

    x = tf.keras.layers.UpSampling2D(size=(resize_factor_0[0], resize_factor_0[1]))(decoding_4_out)
    out = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if 'nadam' in optimizer:
        optimizer = tf.keras.optimizers.Nadam(lr=learning_rate)
    elif 'sgd' in optimizer:
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    elif 'adam' in optimizer:
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss = rmse_coef_slicing
    metric = rmse_coef_slicing

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])  # WORKS

    return model


def mos_binary_classification(input_size=(160, 32), optimizer='adam', learning_rate=1e-5, pooling_factor=[2, 2]):

    def convolutional_block_2(input_layer, n_filters, stride=3):

        x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(input_layer)
        x = tf.keras.layers.Conv2D(n_filters, stride, activation='relu', padding='same')(x)

        return x

    inputs = tf.keras.Input((input_size[0], input_size[1], 1))
    x = convolutional_block_2(inputs, 16)
    x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(x)
    x = convolutional_block_2(x, 32)
    x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(x)
    x = convolutional_block_2(x, 64)
    x = tf.keras.layers.AveragePooling2D(pool_size=(pooling_factor[0], pooling_factor[1]))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    if 'nadam' in optimizer:
        optimizer = tf.keras.optimizers.Nadam(lr=learning_rate)
    elif 'sgd' in optimizer:
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    elif 'adam' in optimizer:
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss = tf.keras.losses.binary_crossentropy
    metric = tf.keras.metrics.binary_accuracy

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])  # WORKS

    return model

############################################
# DIGITAL SIGNAL PROCESSING
############################################

def writewav(f, sr, x):

    xdesn=2**15*(x/np.max(np.abs(x)))
    xint16=xdesn.astype('int16')
    wavfile.write(f, sr, xint16)


def readwav(f):
    fs1, x1 = wavfile.read(f)
    xx1 = x1.astype('float64')
    xx1 *= 2 ** (-15)
    return fs1, xx1


def norm_spectrogram(stft, w_size):
    MD = 80
    ep = 10**(-1*MD/20)

    stft_norm = 20*np.log10(np.abs(stft)/(w_size/2)+ep)
    stft_norm = stft_norm+MD
    stft_norm = stft_norm/MD
    stft_norm = np.round(stft_norm*255)/255

    return stft_norm


def denorm_spectrogram(stft_norm, w_size):
    MD = 80
    ep = 10**(-1*MD/20)

    stft = stft_norm*MD
    stft = stft-MD
    stft = (10**(stft/20)-ep)*(w_size/2)

    return stft


def spectrogram(s, fs):
    s = s.astype('float32')
    w_size_ms = 19.90
    w_size_n = int(w_size_ms * (1e-3) * fs)

    stft = librosa.core.stft(s, n_fft = w_size_n, hop_length=None, win_length=None, window='hann',
                             center=True)

    stft_m = norm_spectrogram(stft, w_size_n)
    stft_p = np.angle(stft)

    return stft_m, stft_p


def recover_from_spectrogram(stft_m, stft_p, fs):
    w_size_ms = 19.90
    w_size_n = int(w_size_ms * (1e-3) * fs)

    # Inverse operations to denormalize stft
    stft_m = denorm_spectrogram(stft_m, w_size_n)
    # Combining phase and signal
    stft = stft_m.astype(np.complex) * np.exp(1j * stft_p)
    # IFFT
    s = librosa.istft(stft, hop_length=None, win_length =None)

    return s

############################################
# EVALUATING, TESTING
############################################


def predict_dataset(model, data_generator, dir_out):

    files = data_generator.files
    fs = 16000

    for i in range(0, len(data_generator)):
        print('File ' + str(i+1) + '/' + str(len(data_generator)))

        batch = data_generator.__getitem__(i)

        farend_speech = np.squeeze(batch[0][0, :, 0])
        nearend_mic = np.squeeze(batch[0][0, :, 1])
        nearend_speech = np.squeeze(batch[1][0, :])

        pred = aec_real_time(model, farend_speech, nearend_mic, fs)

        writewav(dir_out + 'predicted_' + files[i], fs, pred)

    return None


def aec_real_time(model, farend_speech, nearend_mic, fs):

    # Buffer length in ms converted to samples
    n0 = int(120 * (1e-3) * fs)
    n_prev = int(115 * (1e-3) * fs)
    n_post = int(40 * (1e-3) * fs)

    # Loop to load images
    pred = []
    el_time = []
    n_buffers = int(farend_speech.shape[0] / n_post) + 1

    # Initial zero-padding for first buffer
    farend_speech = np.concatenate((np.zeros(n_prev), farend_speech))
    nearend_mic = np.concatenate((np.zeros(n_prev), nearend_mic))

    for iBuffer in range(0, n_buffers):
        # Timer
        t = time.time()

        # Get buffers
        trama_farend_speech = farend_speech[n0 + iBuffer * n_post - n_prev:n0 + (iBuffer + 1) * (n_post)]
        trama_nearend_mic = nearend_mic[n0 + iBuffer * n_post - n_prev:n0 + (iBuffer + 1) * (n_post)]

        # Spectrogram
        stft_farend_speech, _ = spectrogram(trama_farend_speech, fs)
        stft_nearend_mic, stft_nearend_mic_p = spectrogram(trama_nearend_mic, fs)

        # Reshape to create volume
        stft_farend_speech = np.reshape(stft_farend_speech, (stft_farend_speech.shape[0], stft_farend_speech.shape[1], 1))
        stft_nearend_mic = np.reshape(stft_nearend_mic, (stft_nearend_mic.shape[0], stft_nearend_mic.shape[1], 1))

        # Concatenate
        input = np.concatenate([stft_farend_speech, stft_nearend_mic], axis=2)
        input = np.reshape(input, (1, input.shape[0], input.shape[1], input.shape[2]))

        pred_trama = np.squeeze(model.predict(input))
        pred_trama = recover_from_spectrogram(pred_trama, stft_nearend_mic_p, fs)
        pred_trama = np.reshape(pred_trama, (1, pred_trama.shape[0]))

        pred.append(pred_trama[:, -n_post:])

        elapsed = time.time() - t
        el_time.append(elapsed)

    pred = np.concatenate(pred, 1)
    pred = np.squeeze(pred)

    print( 'Elapsed time per buffer: ' + str(np.round(np.average(elapsed) * 1e3, 2)) + ' ms')

    return pred


def predict_dataset_mos_cnn(model, data_generator, dir_out):

    files = data_generator.files
    fs = 16000

    Yref = []
    Ypred = []

    for i in range(0, len(data_generator)):
        print('File ' + str(i+1) + '/' + str(len(data_generator)))

        batch = data_generator.__getitem__(i)

        y = batch[1].item()
        nearend_mic = np.squeeze(batch[0])

        pred = signal_mos_evaluation(model, nearend_mic, fs)

        Yref.append(y)
        Ypred.append(pred)

    df = pd.DataFrame(list(zip(files, Yref, Ypred)),
                      columns=['file_name', 'gt', 'prediction'])
    df.to_excel(dir_out)

    return None


def signal_mos_evaluation(model, nearend_mic, fs):

    # Buffer length in ms converted to samples
    n0 = int(120 * (1e-3) * fs)
    n_prev = int(115 * (1e-3) * fs)
    n_post = int(40 * (1e-3) * fs)

    # Loop to load images
    pred = []
    el_time = []
    n_buffers = int(nearend_mic.shape[0] / n_post) + 1

    # Initial zero-padding for first buffer
    nearend_mic = np.concatenate((np.zeros(n_prev), nearend_mic))

    for iBuffer in range(0, n_buffers-2):
        # Get buffers
        trama_nearend_mic = nearend_mic[n0 + iBuffer * n_post - n_prev:n0 + (iBuffer + 1) * (n_post)]

        # Spectrogram
        stft_nearend_mic, stft_nearend_mic_p = spectrogram(trama_nearend_mic, fs)

        # Reshape to create volume
        stft_nearend_mic = np.reshape(stft_nearend_mic, (stft_nearend_mic.shape[0], stft_nearend_mic.shape[1], 1))
        input = stft_nearend_mic

        # Concatenate
        input = np.reshape(input, (1, input.shape[0], input.shape[1], input.shape[2]))

        pred_trama = np.squeeze(model.predict(input)).item()

        pred.append(pred_trama)

    pred = np.array(pred)
    pred = np.mean(pred)

    return pred

############################################
# DATASET PARTITIONS
############################################

def partition_synthetic_dataset():

    dir_data = '../data/synthetic/farend_speech'
    dir_out = '../data/partitions'

    files = os.listdir(dir_data)
    files = [iFile.split('_')[-1] for iFile in files]

    idx = list(np.arange(0, len(files)))
    idx_test = random.sample(idx, len(idx)//10)

    partition = []
    for i in range(0, len(files)):
        if i in idx_test:
            partition.append('test')
        else:
            partition.append('train')

    df = pd.DataFrame(list(zip(files, partition)),
                      columns=['signal_name', 'partition'])
    df.to_excel(dir_out + '/partition_synthetic.xlsx')


def partition_test_dataset():

    dir_data_clean = '../data/test_set/clean'
    dir_data_noisy = '../data/test_set/noisy'
    dir_out = '../data/partitions'

    files = os.listdir(dir_data_clean)
    files_clean = []
    for iFile in files:
        if iFile.split('_')[-1] == 'c.wav':
            files_clean.append(dir_data_clean + '/' + iFile[0:-10])
    gt_clean = list(np.round(np.ones(len(files_clean))))

    files = os.listdir(dir_data_noisy)
    files_noisy = []
    for iFile in files:
        if iFile.split('_')[-1] == 'mic.wav':
            files_noisy.append(dir_data_noisy + '/' + iFile[0:-8])
    gt_noisy = list(np.round(np.zeros(len(files_clean))))

    files = files_clean + files_noisy
    gt = gt_clean + gt_noisy

    idx = list(np.arange(0, len(files)))
    idx_test = random.sample(idx, len(idx)//10)

    partition = []
    for i in range(0, len(files)):
        if i in idx_test:
            partition.append('test')
        else:
            partition.append('train')

    df = pd.DataFrame(list(zip(files, gt, partition)),
                      columns=['signal_name', 'gt', 'partition'])
    df.to_excel(dir_out + '/partition_test_dataset.xlsx')