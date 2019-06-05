#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:45:17 2019

@author: lagerwer
"""

import numpy as np

# %% Function to compute the FDK weighting
def FDK_weighting(detecpix, det_space, w_detu, w_detv, src_rad, det_rad=0):
    midu = int(detecpix[0]/2)
    midv = int(detecpix[1]/2)
    rho = src_rad + det_rad
    w_FDK = np.ones((np.shape(det_space))) * rho ** 2
    for i in range(np.size(w_FDK, 0)):
        w_FDK[i, :] += ((-midu + i + 1/2) * w_detu) ** 2

    for j in range(np.size(w_FDK, 1)):
        w_FDK[:, j] += ((-midv + j + 1/2) * w_detv) ** 2

    return rho * np.sqrt(1 / w_FDK)


# %% Ramp filter
def ramp_filt(rs_detu):
    mid_det = int(rs_detu / 2)
    filt_impr = np.zeros(mid_det + 1)
    n = np.arange(mid_det + 1)
    tau = 1
    filt_impr[0] = 1 / (4* tau **2)
    filt_impr[1::2] = -1 / (np.pi ** 2 * n[1::2] ** 2 * tau **2)
    filt_impr = np.append(filt_impr[:-1], np.flip(filt_impr[1:], 0))
    return filt_impr


# %%
