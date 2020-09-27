from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(42)

# INPUTS
dir_data = '../data/test_set'
dir_dataframe = '../data/partitions/partition_test_dataset.xlsx'
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
data_generator_tr = DataGeneratorTestDataset(dir_data=dir_data, dataframe=df[df['partition'] == 'train'], batch_size=1, shuffle=True, mode='train')
data_generator_val =DataGeneratorTestDataset(dir_data=dir_data, dataframe=df[df['partition'] == 'test'], batch_size=1, shuffle=False, mode='train')

model = mos_binary_classification(input_size=(160, 32), optimizer='adam', learning_rate=learning_rate,
                                  pooling_factor=[2, 2])


model.fit_generator(data_generator_tr, epochs=100, validation_data=data_generator_val)
model.save('../data/models/model_cnn_mos.h5')