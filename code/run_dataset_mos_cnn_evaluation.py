from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

np.random.seed(42)

# INPUTS
dir_data = '../data/test_set'
dir_dataframe = '../data/partitions/partition_test_dataset.xlsx'
dir_out = '../data/predictions/res_real_dataset.xlsx'

batch_size = 1
fs = 16000
learning_rate = 1e-4

#################################
# TRAIN
#################################

# dataframe
df = pd.read_excel(dir_dataframe)

# Initialize data generator
data_generator_val =DataGeneratorTestDataset(dir_data=dir_data, dataframe=df[df['partition'] == 'test'], batch_size=1, shuffle=False, mode='test')

model = mos_binary_classification(input_size=(160, 32), optimizer='adam', learning_rate=learning_rate,
                                  pooling_factor=[2, 2])
model.load_weights('../data/models/model_cnn_mos.h5')

# Evaluation
predict_dataset_mos_cnn(model, data_generator_val, dir_out)