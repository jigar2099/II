# import lib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class MonteCarlo:
    def __init__(self, input_uniq_shapes, photon_per_samp, required_examples, sample_size):
        '''
        input_uniq_shapes-->unique shapes
        photon_per_sample--->desired photon per sample
        required_examples---> number of samples which are created by adding pulses ad random position, and number of pulse would be photon_per_samp
        sample_size---> Required size for training and creating labels.
        '''
        self.input_uniq_shapes = input_uniq_shapes
        self.photon_per_samp = photon_per_samp
        self.required_examples = required_examples
        self.sample_size = sample_size

    def MC2(self):
        arr_x = np.zeros((self.required_examples, self.sample_size))
        arr_y = np.zeros((self.required_examples, self.sample_size))
        for idx, (value_x, value_y) in enumerate(zip(arr_x,arr_y)):
            # shale selection
            select_ind_shapes = np.random.randint(0,len(self.input_uniq_shapes),self.photon_per_samp)
            for i in select_ind_shapes:# use for loop to find peak position
                shape = self.input_uniq_shapes[i]
                shape_l = len(shape)
                peak, _ = find_peaks(shape)
                if peak.shape[0]==0:#peak_find-->[]
                    peK = 0
                else:
                    peK = peak[0]# peak_fihnd-->[3]
                # Range determination
                before_peak = shape[:peK+0]; len_b_p = len(before_peak)-0
                after_peak = shape[peK:]; len_a_p = len(after_peak)
                lower_bound = 0-len(shape)+1#+1 is because we always want to have shape at the end, if its not there then we can have sample withouth pulse
                #upper_bound = sample_size+len(shape)-1
                upper_bound = self.sample_size-1
                get_ind_to_add = np.random.randint(lower_bound,upper_bound,1)
                gita = get_ind_to_add[0]
                if gita<0:#+len_b_p and gita<0:
                    len_part_to_remove = abs(gita)
                    part_to_remove = shape[:len_part_to_remove]
                    part_to_add = shape[len_part_to_remove:]
                    if len_part_to_remove>len_b_p:
                        part_to_remove = shape[:len_part_to_remove]
                        part_to_add = shape[len_part_to_remove:]
                        value_x[:len(part_to_add)]+=part_to_add
                        frac_of_shape = np.trapz(part_to_add)/np.trapz(shape)
                        value_y[0]+=frac_of_shape
                    else:
                        diff = len_b_p-len_part_to_remove
                        value_x[:len(part_to_add)]+=part_to_add
                        frac_of_shape = np.trapz(part_to_add)/np.trapz(shape)
                        value_y[diff]+=frac_of_shape
                elif gita+len(shape)>self.sample_size-1:#-1 is used for last indice from smaple determination
                    diff = abs(self.sample_size-1-gita)+1
                    part_to_add = shape[:diff]; len_part_to_add=len(part_to_add)
                    value_x[gita:]+=part_to_add
                    frac_of_shape = np.trapz(part_to_add)/np.trapz(shape)
                    if gita+len_b_p>self.sample_size-1:
                        value_y[gita+len_b_p-1]+=frac_of_shape
                    else:
                        value_y[gita+len_b_p]+=frac_of_shape
                else:
                    last_ind = gita+len(shape)
                    value_x[gita:last_ind]+=shape
                    label_ind = gita+len_b_p
                    value_y[label_ind]+=1

        return arr_x,arr_y
        