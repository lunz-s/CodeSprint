#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:42:45 2019

@author: lagerwer
"""

import numpy as np
import odl
import pylab

import support_functions as sup
import FDK_astra_backend as FDK_a
# %%
class CCB_CT:
    # %% Initialize the MR-FDK
    def __init__(self, data_obj, data_struct=True, **kwargs):
        # %% Store the input in the object
        # BE FUCKING CONSISTENT WITH RENAMING ETC. 
        self.pix = data_obj.voxels[0]
        self.angles = data_obj.angles
        self.src_rad = data_obj.src_rad
        self.det_rad = data_obj.det_rad
        self.magn = self.src_rad / (self.src_rad + self.det_rad)
        self.data_struct = data_struct
        self.rec_methods = []

        # %% Set up the geometry
        self.phantom = data_obj
        # Make the reconstruction space
        self.reco_space = self.phantom.reco_space
        voxels = self.phantom.voxels
        factor = 2
        dpix = [int(factor * voxels[0]), voxels[1]]
        self.w_detu = (2 * self.phantom.detecsize[0]) / dpix[0]
        self.w_detv = (2 * self.phantom.detecsize[1]) / dpix[1]
        if self.phantom.data_type == 'simulated':
            self.PH = data_obj.PH
            self.noise = data_obj.noise
            # Do we need a mask?
            src_radius = self.src_rad * self.phantom.volumesize[0] * 2
            det_radius = self.det_rad * self.phantom.volumesize[0] * 2
            # Make a circular scanning geometry
            angle_partition = odl.uniform_partition(0, 2 * np.pi, self.angles)
            self.angle_space = odl.uniform_discr_frompartition(angle_partition)
            # Make a flat detector space
            det_partition = odl.uniform_partition(-self.phantom.detecsize,
                                                   self.phantom.detecsize, dpix)
            self.det_space = odl.uniform_discr_frompartition(det_partition,
                                                         dtype='float32')
            # Create geometry
            self.geometry = odl.tomo.ConeFlatGeometry(
                angle_partition, det_partition, src_radius=src_radius,
                                det_radius=det_radius, axis=[0, 0, 1])
        else:
            src_radius = self.phantom.src_rad
            det_radius = self.phantom.det_rad
            self.pix_size = self.phantom.pix_size
            self.angle_space = self.phantom.angle_space
            self.angles = np.size(self.angle_space)
            self.det_space = self.phantom.det_space
            self.geometry = self.phantom.geometry

        # %% Create the FP and BP and the data
        # Forward Projection
        self.FwP = odl.tomo.RayTransform(self.reco_space, self.geometry,
                                         use_cache=False)
        self.rs_detu = int(2 ** (np.ceil(np.log2(dpix[0])) + 1))
        self.g = self.phantom.g

    def FDK_filt(self, filt_type):
        filt = np.real(np.fft.rfft(sup.ramp_filt(self.rs_detu)))
        freq = 2 * np.arange(len(filt))/(self.rs_detu)
        if filt_type == 'Ram-Lak':
            pass
        elif filt_type == 'Shepp-Logan':
            filt = filt * np.sinc(freq / 2)
        elif filt_type == 'Cosine':
            filt = filt * np.cos(freq * np.pi / 2)
        elif filt_type == 'Hamming':
            filt = filt * (0.54 + 0.46 * np.cos(freq * np.pi))
        elif filt_type == 'Hann':
            filt = filt * (np.cos(freq * np.pi / 2) ** 2)
        else:
            raise ValueError('unknown `filter_type` ({})'
                         ''.format(filt_type))
        weight = 1 / 2 / self.w_detu
        return (filt * weight)


# %% Initialize the algorithms
    def do_FDK(self, astra=True):
        hf = self.FDK_filt('Ram-Lak')
        return self.reco_space.element(
                FDK_a.FDK_astra(self.g, hf, self.geometry, self.reco_space))

        

# %% Algorithm support functions
    def pd_FFT(self, h):
            return self.fourier_filter_space.element(np.fft.rfft(
                    np.fft.ifftshift(self.conv_op.rs_filt(h))))

    def pd_IFFT(self, hf):
        return self.conv_op.rs_filt.inverse(np.fft.fftshift(np.fft.irfft(hf)))


# %%
    def show_phantom(self, clim=None, save_name=None, extension='.eps',
                     fontsize=20):
        space = self.reco_space
        if clim == None:
            clim = [np.min(self.phantom.f),
                    np.max(self.phantom.f)]
        mid = np.shape(self.phantom.f)[0] // 2
        xy = self.phantom.f[:, :, mid]
        xz = self.phantom.f[:, mid, :]
        yz = self.phantom.f[mid, :, :]
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ima = ax1.imshow(np.rot90(xy), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[1], space.max_pt[1]],
                            cmap='gray')
        ax1.set_xlabel('x', fontsize=fontsize)
        ax1.set_ylabel('y', fontsize=fontsize)
        ima = ax2.imshow(np.rot90(xz), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax2.set_xlabel('x', fontsize=fontsize)
        ax2.set_ylabel('z', fontsize=fontsize)
        ima = ax3.imshow(np.rot90(yz), clim=clim, extent=[space.min_pt[1],
                         space.max_pt[1],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax3.set_xlabel('y', fontsize=fontsize)
        ax3.set_ylabel('z', fontsize=fontsize)
        fig.colorbar(ima, ax=(ax1, ax2, ax3))
        if save_name is None:
            if self.phantom.data_type == 'simulated':
                fig.suptitle('Ground truth', fontsize=fontsize+2)
            else:
                fig.suptitle('Gold standard', fontsize=fontsize+2)
        fig.show()
        if save_name is not None:
            pylab.savefig(save_name+extension, bbox_inches='tight')
            
            
# %%
    def show(self, rec, clim=None, save_name=None, extension='.eps',
             fontsize=20):
        space = self.reco_space
        mid = np.size(space, 0) // 2
        if clim == None:
            clim = [np.min(self.phantom.f),
                    np.max(self.phantom.f)]
        xy, xz, yz = [rec[:, :, mid], rec[:, mid, :], rec[mid, :, :]]
        fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=[20, 6])
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ima = ax1.imshow(np.rot90(xy), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[1], space.max_pt[1]],
                            cmap='gray')
        ax1.set_xticks([],[])
        ax1.set_yticks([],[])
        ax1.set_xlabel('x', fontsize=fontsize)
        ax1.set_ylabel('y', fontsize=fontsize)
        ima = ax2.imshow(np.rot90(xz), clim=clim, extent=[space.min_pt[0],
                         space.max_pt[0],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax2.set_xticks([],[])
        ax2.set_yticks([],[])
        ax2.set_xlabel('x', fontsize=fontsize)
        ax2.set_ylabel('z', fontsize=fontsize)
        ima = ax3.imshow(np.rot90(yz), clim=clim, extent=[space.min_pt[1],
                         space.max_pt[1],space.min_pt[2], space.max_pt[2]],
                            cmap='gray')
        ax3.set_xlabel('y', fontsize=fontsize)
        ax3.set_ylabel('z', fontsize=fontsize)
        ax3.set_xticks([],[])
        ax3.set_yticks([],[])
        fig.colorbar(ima, ax=(ax1, ax2, ax3))

        fig.suptitle('Reconstruction', fontsize=fontsize+2)
        fig.show()
