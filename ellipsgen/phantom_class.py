#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:25:07 2019

@author: lagerwer
"""
import numpy as np
import odl
import gc 

def make_ellipsoids(number):
    ellipsoids = []
    for i in range(number):
        value = np.random.rand(1)
        axis = np.random.rand(3)
        centers = np.random.rand(3) * 2 - 1
        rotations = np.random.rand(3) * 2 * np.pi
        ellipsoids += [[*value, *axis, *centers, *rotations]]
    return ellipsoids
# %%

class phantom:
    def __init__(self, voxels, PH, angles, noise, src_rad, det_rad, **kwargs):
        self.data_type = 'simulated'
        voxels_up = [int(v * 1) for v in voxels]
        self.voxels = voxels
        self.PH = PH
        self.angles = angles
        self.noise = noise
        self.src_rad = src_rad
        self.det_rad = det_rad
        if self.PH == 'Zero' and self.noise == None:
            raise ValueError('Specify a noise level for this phantom')
            
        if 'load_data_g' in kwargs:
            if PH in ['Fourshape', 'Threeshape', '22 Ellipses']:
                if 'load_data_f' not in kwargs:
                    raise ValueError('If you load data for this phantom,' + 
                                     ' you also need to provide a phantom')
            self.reco_space, self.f = self.phantom_creation(voxels, **kwargs)
            self.generate_data(voxels, self.reco_space, self.f, **kwargs)
        else:
            reco_space_up, f_up = self.phantom_creation(voxels_up, **kwargs)
            self.generate_data(voxels_up, reco_space_up, f_up, **kwargs)
            reco_space_up, f_up = None, None
            gc.collect()
            self.reco_space, self.f = self.phantom_creation(voxels, 
                                                            second=True,
                                                            **kwargs)
    
# %%    
    def generate_data(self, voxels_up, reco_space_up, f_up, **kwargs):
        factor = 2
        dpix_up = [factor * voxels_up[0], voxels_up[1]]
        dpix = [int(factor * self.voxels[0]), self.voxels[0]]
        src_radius = self.src_rad * self.volumesize[0] * 2
        det_radius = self.det_rad * self.volumesize[0] * 2
        # Make a circular scanning geometry
        angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
        # Make a flat detector space
        det_partition = odl.uniform_partition(-self.detecsize,
                                               self.detecsize, dpix_up)
        # Create data_space_up and data_space
        data_space = odl.uniform_discr((0, *-self.detecsize),
                                       (2 * np.pi, *self.detecsize),
                                       [self.angles, *dpix], dtype='float32')
        data_space_up = odl.uniform_discr((0, *-self.detecsize),
                                       (2 * np.pi, *self.detecsize), 
                                       [self.angles, *dpix_up], dtype='float32')
        # Create geometry
        geometry = odl.tomo.ConeFlatGeometry(
            angle_partition, det_partition, src_radius=src_radius,
                            det_radius=det_radius, axis=[0, 0, 1])

        FP = odl.tomo.RayTransform(reco_space_up, geometry, use_cache=False)

        resamp = odl.Resampling(data_space_up, data_space)
        if 'load_data_g' in kwargs:
            if type(kwargs['load_data_g']) == str: 
                self.g = data_space.element(np.load(kwargs['load_data_g']))
            else:
                self.g = data_space.element(kwargs['load_data_g'])
        else:
            self.g = resamp(FP(f_up))

            if self.noise == None:
                pass
            elif self.noise[0] == 'Gaussian':
                self.g += data_space.element(
                        odl.phantom.white_noise(resamp.range) * \
                        np.mean(self.g) * self.noise[1])
            elif self.noise[0] == 'Poisson':
                # 2**8 seems to be the minimal accepted I_0
                self.g = data_space.element(
                        self.add_poisson_noise(self.noise[1]))
            else:
                raise ValueError('unknown `noise type` ({})'
                             ''.format(self.noise[0]))

# %%
    def phantom_creation(self, voxels, **kwargs):
        if self.PH == '22 Ellipses':
            self.volumesize = np.array([4, 4, 4], dtype='float32')
            # Scale the detector correctly
            self.detecsize = np.array([2 * self.volumesize[0],
                                       self.volumesize[1]])
            # Make the reconstruction space
            reco_space = odl.uniform_discr(min_pt=-self.volumesize,
                                           max_pt=self.volumesize,
                                            shape=voxels, dtype='float32')

            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                if 'second' in kwargs:
                    seed_old = np.random.get_state()
                    np.random.set_state(self.seed)
                else:
                    self.seed = np.random.get_state()
            if 'load_data_f' in kwargs:
                f = reco_space.element(np.load(kwargs['load_data_f']))
            else:
                ellipsoids = make_ellipsoids(22)
                phantom = odl.phantom.ellipsoid_phantom(reco_space,
                                                        ellipsoids)
                f = phantom / np.max(phantom) * .22
            if 'second' in kwargs:
                np.random.set_state(seed_old)
            return reco_space, f 
        else:
            raise ValueError('unknown `Phantom name` ({})'
                             ''.format(self.PH))
        

# %%           
    def add_poisson_noise(self, I_0, seed=None):
        seed_old = np.random.get_state()
        np.random.seed(seed)
        data = np.asarray(self.g.copy())
        Iclean = (I_0 * np.exp(-data))
        data = None
        Inoise = np.random.poisson(Iclean)
        Iclean = None
        np.random.set_state(seed_old)
        return  (-np.log(Inoise / I_0))


# %%
        