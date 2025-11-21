import os
import glob
import numpy as np
from data_schemas import SASDataSchema, SysParams, WfmParams
import constants as c
import scipy.io
import pickle
from geometry import create_voxels
from sas_utils import crop_wfm

def process_folder():
    # Coordinates.csv contains the coordinates for each flight (Flight #, Angle, Height, Temp, Waveform)
    # SysParams.csv contains system parameters. (Speaker xyz, Mic 1 xyz, Mic 2 xyz, Center xyz, Fs, GD, Other stuff
    # WaveformParams.csv contains waveform parameters
    # Waveforms.csv contains the waveform

    # wfm_data = scipy.io.loadmat('./rawData_matlab/new_pyramid_point_sphere.mat')
    tx_coords = scipy.io.loadmat('./rawData_matlab/tx_coord_r230_1.mat')
    # tx_coords = scipy.io.loadmat('./rawData_matlab/tx_coord_sphere.mat')

    # wfm_data = wfm_data['rawDataCal']
    wfm_data = np.load('./dataset_creation/simulated_data/simulated_sar_data_bunny_2_4e9.npy')

    tx_coords = tx_coords['tx_coords']

    print(wfm_data.shape)
    print(tx_coords.shape)

    wfm_data = np.array(wfm_data)
    tx_coords = np.array(tx_coords)
    tx_coords[:, [1, 2]] = tx_coords[:, [2, 1]]

    print(wfm_data.shape)
    wfm_data = wfm_data.reshape(691200, 256, order='F')
    # wfm_data = wfm_data.reshape(700000, 256, order='F')
    print(wfm_data.shape)

    sys_params = SysParams()
    sys_params[c.TX_POS] = np.array([ 0, -0.23,  0])
    sys_params[c.RX_POS] = np.array([ 0, -0.23,  0])
    sys_params[c.CENTER] = np.array([0,0,0])
    # sys_params[c.GROUP_DELAY] = group_delay
    sys_params[c.FS] = 5000000

    wfm_params = WfmParams()
    # wfm_params[c.F_START] = f_start
    # wfm_params[c.F_STOP] = f_stop
    # wfm_params[c.T_DUR] = t_dur
    # wfm_params[c.WIN_RATIO] = win_ratio

    # wfm = []
    # wfm = np.array(wfm)

    dim = 0.18

    geometry = create_voxels(-dim, dim,
                             -dim, dim,
                             -dim, dim,
                             150, 150, 150)

    airsas_data = SASDataSchema()
    airsas_data[c.WFM_DATA] = wfm_data
    airsas_data[c.TX_COORDS] = tx_coords
    airsas_data[c.RX_COORDS] = tx_coords
    # airsas_data[c.TEMPS] = temps
    airsas_data[c.SYS_PARAMS] = sys_params
    airsas_data[c.WFM_PARAMS] = wfm_params
    airsas_data[c.GEOMETRY] = geometry
    # airsas_data[c.WFM] = wfm


    speed_of_sound = 299792458
    wfm_length = 256
    K = 0.70295e14

    wfm_crop_settings = crop_wfm(airsas_data[c.TX_COORDS],
                                 airsas_data[c.RX_COORDS],
                                 geometry[c.CORNERS],
                                 wfm_length,
                                 airsas_data[c.SYS_PARAMS][c.FS],
                                 K,
                                 speed_of_sound)
    
    airsas_data[c.WFM_CROP_SETTINGS] = wfm_crop_settings

    print("in process")

    with open(os.path.join('output_dir', 'sim_data_run_bunny_2_4e9_cylinder.pik'), 'wb') as handle:
        print("Saving system data to ", os.path.join('output_dir', 'sim_data_run_bunny_2_4e9_cylinder.pik'))
        pickle.dump(airsas_data, handle)


    return airsas_data


if __name__ == '__main__':
    print("in main")
    airsas_data = process_folder()