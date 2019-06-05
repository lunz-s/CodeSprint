#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:31:24 2019

@author: lagerwer
"""

import phantom_class as ph
import CCB_class as CT
import numpy as np
import astra
# %%
num_vox = 256
voxels = [num_vox, num_vox, num_vox]
angles = 360
noise = None #['Poisson', 2 ** 14]
src_rad = 10
det_rad = 0
PH = '22 Ellipses'


data_obj = ph.phantom(voxels, PH, angles, noise, src_rad, det_rad)
# %%
case = CT.CCB_CT(data_obj)

# %%
rec = case.do_FDK()
# %%
case.show_phantom()
case.show(rec)
