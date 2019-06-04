#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:02:44 2019

@author: lagerwer
"""

import os
import CCB_class as CT
import numpy as np
import phantom_class as ph


def number_of_datasets(path):
    if os.path.exists(path  + '0.npy'):
        nDatasets = 1
        while os.path.exists(path + str(nDatasets) + '.npy'):
            nDatasets += 1
    else:
        nDatasets = 0
    return nDatasets


def generate_data(path, num_datasets, num_vox, num_angles, noise):
    path_GT = path + '/GT/dataset'
    path_FDK = path + '/FDK/dataset'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + '/GT')
        os.makedirs(path + '/FDK')
    
    nData = number_of_datasets(path_GT)
    if nData < num_datasets:
        for i in range(num_datasets - nData):
            data_obj = ph.phantom(voxels, '22 Ellipses', num_angles, noise, 
                              src_rad, det_rad)
            np.save(path_GT + str(i), data_obj.f)
            case = CT.CCB_CT(data_obj)
            rec = case.do_FDK()
            np.save(path_FDK + str(i), rec)
            print('Finsihed making dataset pair ' + str(i))

            

        
        
        
    
