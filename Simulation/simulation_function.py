import sys
import os
import numpy as np
#import matplotlib.pyplot as plt
import math
import time
import SimpleITK as sitk

current_path = os.path.dirname(__file__)
sys.path.append(current_path+'/help_function')
sys.path.append(current_path+'/kwave_function')

import make_transducer as hlp
import skull_processing_vertical as sp_vertical

from kwave_input_file import KWaveInputFile
from kwave_output_file import KWaveOutputFile, DomainSamplingType, SensorSamplingType
from kwave_data_filters import SpectralDataFilter
from kwave_bin_driver import KWaveBinaryDriver

l2n = lambda l: np.array(l)
n2l = lambda n: list(n)
start = time.time()

class makeSimulation():

    def __init__(self):

        ####################################################################
        # Material properties
        self.c_water = 1482  # [m/s]
        self.d_water = 1000  # [kg/m^3]
        self.a_water = 3.84e-4  # [Np/MHz/cm]

        self.c_bone = 2800  # [m/s]    # 2800 or 3100 m/s
        self.d_bone = 2200  # [kg/m^3]
        self.a_bone_min = 21.5  # [Np/MHz/cm]
        self.a_bone_max = 208.9  # [Np/MHz/cm]

        self.alpha_power = 1.01

        ####################################################################
        # Source properties
        self.amplitude = 1  # source pressure [Pa]
        self.source_freq = 2e5  # frequency [Hz]
        self.ROC = 116  # [mm]     # transducer setting
        self.width = 60  # [mm]

        # Bounary condition
        self.boundary = 0

        # Time step
        self.CFL = 0.1
        self.end_time = 150e-6
        self.points_per_wavelength = 2*np.pi  # number of grid points per wavelength at f0


    def preprocessing(self, itk_image, tran_pose, target_pose):

        ####################################################################
        # Source properties
        c_water = self.c_water      # [m/s]

        ####################################################################
        # Grid properties
        points_per_wavelength = self.points_per_wavelength    # number of grid points per wavelength at f0
        source_freq = self.source_freq
        ROC = self.ROC
        width = self.width


        dx = c_water / (points_per_wavelength * source_freq) # [m]
        dy = dx
        dz = dx

        ####################################################################
        # Grid_size contains the PML (default 20)
        grid_res = (dx, dy, dz)

        # Time step
        CFL      = self.CFL
        end_time = self.end_time
        dt       = CFL * grid_res[0] / c_water
        steps    = int(end_time / dt)


        ####################################################################
        # Skull process
        tran_pose = np.multiply(tran_pose,  (-1, -1, 1))
        target_pose = np.multiply(target_pose,  (-1, -1, 1))
        print("Perform skull processing")
        skullCrop_arr, skullCrop_itk, image_cordi = sp_vertical.make_skull_medium_vertical(itk_image,  l2n(grid_res)*1000, tran_pose, target_pose, self.boundary)


        ####################################################################
        # Transducer signal
        p0, trans_itk = hlp.makeTransducer_vertical(image_cordi, dx*1e3, ROC, width, tran_pose, target_pose)

        return skullCrop_arr, skullCrop_itk, image_cordi, p0, trans_itk


    def run_simulation(self, skullCrop_arr, image_cordi, p0, model_dir):
        ####################################################################
        # Input, Output name
        input_filename  = model_dir+'/kwave_in.h5'
        output_filename = model_dir+'/kwave_out.h5'


        ####################################################################
        # Source properties
        amplitude = self.amplitude       # source pressure [Pa]
        source_freq = self.source_freq     # frequency [Hz]


        ####################################################################
        # Material properties
        c_water = self.c_water      # [m/s]
        d_water = self.d_water      # [kg/m^3]
        a_water = self.a_water   # [Np/MHz/cm]

        c_bone = self.c_bone       # [m/s]    # 2800 or 3100 m/s
        d_bone = self.d_bone       # [kg/m^3]
        a_bone_min = self.a_bone_min   # [Np/MHz/cm]
        a_bone_max = self.a_bone_max  # [Np/MHz/cm]

        alpha_power = self.alpha_power

        # ref by: Marquet F, Pernot M, Aubry J F, Montaldo G, Marsac L, Tanter M and Fink M 2009 Non-invasive transcranial
        # ultrasound therapy based on a 3D CT scan: protocol validation and in vitro results

        ####################################################################
        # Grid properties
        points_per_wavelength = self.points_per_wavelength    # number of grid points per wavelength at f0

        dx = c_water / (points_per_wavelength * source_freq) # [m]
        dy = dx
        dz = dx

        ####################################################################
        # Grid_size contains the PML (default 20)
        grid_res = (dx, dy, dz)

        # Time step
        CFL      = self.CFL
        end_time = self.end_time
        dt       = CFL * grid_res[0] / c_water
        steps    = int(end_time / dt)


        ####################################################################
        # Skull process
        print("## Perform skull processing")
        grid_size = skullCrop_arr.shape

        # normalize HU value
        skullCrop_arr[skullCrop_arr > 3000] = 3000
        skullCrop_arr[skullCrop_arr < 220 ] = 0
        print("## Finish voxelization")


        ####################################################################
        # assign skull properties depend on HU value  - Ref. Numerical evaluation, Muler et al, 2017
        # PI = 1 - (skull_arr/1000)
        # ct_sound_speed = c_water*PI + c_bone*(1-PI)
        # ct_density  = d_water*PI + d_bone*(1-PI)
        #
        # ct_att          = a_bone_min + (a_bone_max-a_bone_min)*np.power(PI, 0.5)
        # ct_att[PI==1]   = a_water


        ####################################################################
        # assign skull properties depend on HU value  - Ref. Multi resolution, Yoon et al, 2019
        PI = skullCrop_arr/np.max(skullCrop_arr)
        ct_sound_speed = c_water + (c_bone - c_water)*PI
        ct_density     = d_water + (d_bone - d_water)*PI
        ct_att         = a_water + (a_bone_min - a_water)*PI


        ####################################################################
        # Assign material properties
        sound_speed     = ct_sound_speed
        density         = ct_density
        alpha_coeff_np  = ct_att


        alpha_coeff = hlp.neper2db(alpha_coeff_np*source_freq/1e6/pow(2*np.pi*source_freq, alpha_power), alpha_power)


        ####################################################################
        # Define simulation input and output files
        print("## k-wave core input function")
        input_file  = KWaveInputFile(input_filename, grid_size, steps, grid_res, dt)
        output_file = KWaveOutputFile(file_name=output_filename)


        ####################################################################
        # Transducer signal
        source_signal = amplitude * np.sin((2*math.pi)*source_freq*np.arange(0.0, steps*dt, dt))


        ####################################################################
        # Open the simulation input file and fill it as usual
        with input_file as file:
            file.write_medium_sound_speed(sound_speed)
            file.write_medium_density(density)
            file.write_medium_absorbing(alpha_coeff, alpha_power)
            file.write_source_input_p(file.domain_mask_to_index(p0), source_signal, KWaveInputFile.SourceMode.ADDITIVE, c_water)

            sensor_mask = np.ones(grid_size)
            file.write_sensor_mask_index(file.domain_mask_to_index(sensor_mask))

        # Create k-Wave solver driver, which will call C++/CUDA k-Wave binary.
        # It is usually necessary to specify path to the binary: "binary_path=..."
        driver = KWaveBinaryDriver()


        # Specify which data should be sampled during the simulation (final pressure in the domain and
        # RAW pressure at the sensor mask
        driver.store_pressure_everywhere([DomainSamplingType.MAX])

        # Execute the solver with specified input and output files
        driver.run(input_file, output_file)
        print("## Calculation time :", time.time() - start)


        #Open the output file and generate plots from the results
        with output_file as file:

            p_max = file.read_pressure_everywhere(DomainSamplingType.MAX)
            np.save('Result_numpy_ver', p_max)

            sitk_p_max = p_max
            sitk_p_max = sitk_p_max.transpose([2, 1, 0])
            sitk_p_max = sitk_p_max/np.max(sitk_p_max)
            spacing = np.array((dx, dy, dz))*1e3
            origin = np.array((min(image_cordi[0]), min(image_cordi[1]), min(image_cordi[2])))
            result_itk = sitk.GetImageFromArray(sitk_p_max, sitk.sitkInt8)
            result_itk.SetSpacing(spacing)
            result_itk.SetOrigin(origin)


        return result_itk