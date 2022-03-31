# import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from monte_carlo import MonteCarlo as mc

class MC_set:
    def __init__(self, filepath, max_Rate, exe, sample_size):
        self.filepath = np.load(filepath)
        self.max_Rate= max_Rate
        self.exe = exe
        self.sample_size = sample_size
    def remove_duplicate_pulse(self):
        x_all = self.filepath.reshape(int(self.filepath.shape[0]/100),100)
        # Remove duplicates
        shapes = [x_all[i][x_all[i]!=0] for i in range(x_all.shape[0])]# creates list of shapes
        uniq_shapes = [list(j) for j in set(map(tuple,shapes))]# sublist match in the form of tuple
        return uniq_shapes
    def train_test_val_split(self, train, test, val):
        y = self.remove_duplicate_pulse()
        return y[:train], y[:test], y[:val]
        
    def get_mc_set(self, shape_list):
        data_X = []
        data_y = []
        for i in range(self.max_Rate):
            #print('samle_len : ', len(shape_list))
            #print('rate : ', i)
            #print('samp_size : ', self.sample_size)
            X,y = mc(shape_list, i, self.exe, self.sample_size).MC2()
            data_X.append(X)
            data_y.append(y)
        #data_X, data_y = mc(shape_list,2,10,100).MC2()
        data_X = np.array(data_X); data_X = data_X.reshape(self.exe*self.max_Rate, self.sample_size)
        data_y = np.array(data_y); data_y = data_y.reshape(self.exe*self.max_Rate, self.sample_size)
        return data_X, data_y