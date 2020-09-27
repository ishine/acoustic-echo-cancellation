from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(42)

# INPUTS
dir_farend_speech = '../data/synthetic/farend_speech/'
dir_nearend_mic = '../data/synthetic/nearend_mic_signal/'
dir_nearend_speech = '../data/synthetic/nearend_speech/'
dir_dataframe = '../data/partitions/partition_synthetic.xlsx'
dir_predictions = '../data/predictions/synthetic/'

batch_size = 1
fs = 16000
learning_rate = 1e-4

#################################
# TRAIN
#################################

# dataframe
df = pd.read_excel(dir_dataframe)

# Initialize data generator
data_generator_tr = DataGeneratorSyntheticDataset(dataframe=df[df['partition'] == 'train'],
                                                  dir_farend_speech=dir_farend_speech, dir_nearend_mic=dir_nearend_mic,
                                                  dir_nearend_speech=dir_nearend_speech, batch_size=batch_size,
                                                  shuffle=False)
data_generator_val = DataGeneratorSyntheticDataset(dataframe=df[df['partition'] == 'test'],
                                                   dir_farend_speech=dir_farend_speech, dir_nearend_mic=dir_nearend_mic,
                                                   dir_nearend_speech=dir_nearend_speech, batch_size=batch_size,
                                                   shuffle=False)

model = u_net_2d(input_size=(160, 32), optimizer='nadam', learning_rate=learning_rate, mode=2,
                 number_filters_0=64, resize_factor_0=[1, 1])

model.fit_generator( data_generator_tr, epochs=15, validation_data=data_generator_val,)
model.save('../data/models/model_slicing.h5')
