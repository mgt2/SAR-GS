# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import os
# import sys
# from PIL import Image
# from typing import NamedTuple

# import numpy as np
# import json
# from pathlib import Path
# from plyfile import PlyData, PlyElement
# from scene.gaussian_model import BasicPointCloud
# import torch
# import imageio
# import math
# import pandas as pd
# import yaml
# from scipy.spatial.transform import Rotation
# import random

# class SpectrumInfo(NamedTuple):
#     R: np.array           # A NumPy array representing the rotation matrix of the camera.
#     T_rx: np.array        # A NumPy array representing the translation vector of the camera.
#     T_tx: np.array
#     spectrum: np.array    # A NumPy array containing image data.
#     spectrum_path: str    # A string representing the file path to the image.
#     spectrum_name: str    # A string representing the name of the image.
#     width: int     
#     height: int


# class SceneInfo(NamedTuple):
#     point_cloud: BasicPointCloud
#     train_spectrums: list
#     test_spectrums: list
#     nerf_normalization: dict
#     ply_path: str


# def split_dataset_train_v2(datadir, train_path, test_path, ratio=0.8):
#     # here the ratio is the ratio of training set
#     # train and test set ratios are fixed to 0.7 and 0.3
#     llffhold_t = 8

#     spectrum_dir = os.path.join(datadir, 'spectrum')
#     spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
#     image_names = [x.split('.')[0] for x in spt_names]

#     len_image = len(image_names)

#     # Set the seed for reproducibility
#     random.seed(1994)
#     np.random.seed(1994)

#     test_index = np.arange(int(len_image))[:: llffhold_t]
#     train_index_raw = np.array([j for j in np.arange(int(len_image)) if (j not in test_index)])
#     train_len = len(train_index_raw)

#     number_train = int(train_len * ratio)
#     train_index = np.random.choice(train_index_raw, number_train, replace=False)

#     print("\n Ratio in Train set: {}....  Train set: {}....  Test set: {}....\n".format(ratio, number_train, len(test_index)))

#     train_image = [image_names[idx] for idx in train_index]
#     test_image = [image_names[idx] for idx in test_index]

#     np.savetxt(train_path, train_index, fmt='%s')
#     np.savetxt(test_path,  test_index,  fmt='%s')


# def readSpectrumImage(data_dir_path):
#     data_infos = []

#     tx_pos_path = os.path.join(data_dir_path, 'tx_pos.csv')
#     tx_pos = pd.read_csv(tx_pos_path).values               # (N, 3)
#     # tx_pos = torch.tensor(tx_pos, dtype=torch.float32)   # torch.Size([N, 3]), N is N is 6123, torch.Size([6123, 3])

#     gateway_pos_path = os.path.join(data_dir_path, 'gateway_info.yml')
#     spectrum_dir     = os.path.join(data_dir_path, 'spectrum')
#     spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])

#     with open(gateway_pos_path) as f_loader:
#         gateway_info = yaml.safe_load(f_loader)
        
#          # [5.0, 0.26, 0]
#         gateway_pos = gateway_info['gateway1']['position']
#         gateway_quaternion = gateway_info['gateway1']['orientation']

#     for image_idx, image_name in enumerate(spt_names):

#         qvec = np.array(gateway_quaternion)
#         # torch.Size([3, 3])
#         rotation_matrix = torch.from_numpy(Rotation.from_quat(qvec).as_matrix()).float()

#         tvec_rx = torch.from_numpy(np.array(gateway_pos)).float()         # torch.Size([3])

#         tvec_tx = torch.from_numpy(np.array(tx_pos[image_idx])).float()   # torch.Size([3])

#         # extr.name: '00001.jpg', os.path.basename(extr.name): '00001.jpg'
#         image_path = os.path.join(spectrum_dir, os.path.basename(image_name))

#         # os.path.basename(image_path): '00001.jpg', image_name: '00001'
#         image_name_t = os.path.basename(image_path).split(".")[0]

#         # The imread function handles opening and closing the file internally, 
#         #       so once the function call is complete, the file is automatically closed.
#         image = imageio.imread(image_path).astype(np.float32) / 255.0

#         height = image.shape[0]
#         width  = image.shape[1]

#         resized_image = torch.from_numpy(np.array(image)).float()   # torch.Size([90, 360])

#         # resized_image = resized_image.unsqueeze(dim=-1).permute(2, 0, 1).repeat(3, 1, 1)

#         spec_info = SpectrumInfo(R=rotation_matrix, 
#                                 #  T_rx=tvec_tx,
#                                 #  T_tx=tvec_rx,
#                                  T_rx=tvec_rx,
#                                  T_tx=tvec_tx,
#                                  spectrum=resized_image, 
#                                  spectrum_path=image_path, 
#                                  spectrum_name=image_name_t,
#                                  height=height,
#                                  width=width)
        
#         data_infos.append(spec_info)

#     sys.stdout.write('\n')

#     return data_infos


# def getNorm_3d(specs_info, scale):

#     # scale = 2.0
#     # len(cam_centers): 301, each one is an tensor of torch.Size([3])
#     def get_center_and_diag(gatewa_pos_t, cam_center):

#         gatewa_pos_t = gatewa_pos_t.unsqueeze(1)  # torch.Size([3, 1])

#         # torch.Size([3, 6124])
#         cam_center = torch.stack(cam_center, dim=1)

#         dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
#         radius = torch.max(dists) * scale

#         deviations = cam_center - gatewa_pos_t

#         # Clone to avoid modifying original
#         positive_deviations = deviations.clone()
#         negative_deviations = deviations.clone()

#         positive_deviations[positive_deviations < 0] = 0
#         negative_deviations[negative_deviations > 0] = 0

#         max_positive = positive_deviations.max(dim=1).values
#         max_negative = negative_deviations.min(dim=1).values.abs()

#         epsilon = 1e-6
#         max_positive[max_positive < epsilon] = 1.0
#         max_negative[max_negative < epsilon] = 1.0

#         return {"max_positive": max_positive * scale, "max_negative": max_negative * scale}, radius.item()
    
#     cam_centers = []
#     gatewa_pos  = specs_info[0].T_rx   # gateway location, torch.Size([3])
#     # gatewa_pos  = specs_info[0].T_tx   # gateway location, torch.Size([3])


#     for cam in specs_info:
#         cam_centers.append(cam.T_tx)  # TX location, torch.Size([3])
#         # cam_centers.append(cam.T_rx)  # TX location, torch.Size([3])

#     # diagonal: {'max_positive': tensor([1.2000, 0.2820, 2.0412]), 
#     #            'max_negative': tensor([7.0716, 1.3428, 1.2000])}

#     #           {'max_positive': tensor([2.0000, 0.4700, 3.4020]), 
#     #            'max_negative': tensor([11.7860,  2.2380,  2.0000])}

#     # radius: 7.3305511474609375   12.217584609985352
#     diagonal, radius = get_center_and_diag(gatewa_pos, cam_centers)

#     translate = -gatewa_pos   # tensor([-5.0000, -0.2600, -0.0000])  

#     return {"translate": translate, "radius": radius, "extent": diagonal}


# def obtain_train_test_idx(args_model, len_list):

#     path     = args_model.source_path
#     llffhold = args_model.llffhold

#     llffhold_flag = args_model.llffhold_flag

#     train_index = os.path.join(path, args_model.train_index_path)    # train_index_knn.txt
#     test_index  = os.path.join(path, args_model.test_index_path)     # test_index_knn.txt

#     if llffhold_flag:
        
#         print("\nUSING LLFFHOLD INDEX FILE\n")
#         # the index start from 0
#         i_test = np.arange(int(len_list))[:: llffhold]
#         i_train = np.array([j for j in np.arange(int(len_list)) if (j not in i_test)])

#     elif "knn" in train_index:
#         print("\nUSING KNN INDEX FILE\n")
#         # the index start from 0
#         i_train = np.loadtxt(train_index, dtype=int)
#         i_test  = np.loadtxt(test_index,  dtype=int)
        
#     else:
#         print("\nUSING RANDOM INDEX FILE\n")
#         # the index start from 1, if no - 1, since 00001.png is the first image
#         i_train = np.loadtxt(train_index, dtype=int)
#         i_test  = np.loadtxt(test_index,  dtype=int)

#     return i_train, i_test


# def readRFSceneInfo(args_model):

#     path         = args_model.source_path
#     eval         = args_model.eval
#     camera_scale     = args_model.camera_scale
#     voxel_size_scale = args_model.voxel_size_scale

#     ratio_train = args_model.ratio_train

#     spectrums_infos_unsorted = readSpectrumImage(path)

#     train_index_path = os.path.join(path, args_model.train_index_path)    
#     test_index_path  = os.path.join(path, args_model.test_index_path)  

#     split_dataset_train_v2(path, train_index_path, test_index_path, ratio=ratio_train)

#     i_train, i_test = obtain_train_test_idx(args_model, len(spectrums_infos_unsorted))
    
#     spectrums_infos = sorted(spectrums_infos_unsorted.copy(), key = lambda x : int(x.spectrum_name))

#     if eval:
#         train_infos = [spectrums_infos[idx] for idx in i_train]
#         test_infos  = [spectrums_infos[idx] for idx in i_test]

#     else:
#         train_infos = spectrums_infos
#         test_infos = []

#     nerf_normalization = getNorm_3d(spectrums_infos, camera_scale)

#     ply_path = os.path.join(path, "points3D.ply")
#     if ((not os.path.exists(ply_path)) or (args_model.gene_init_point)):

#         receiver_pos = spectrums_infos[0].T_rx.numpy()

#         cube_size = round((3.00e8 / 902.0e6) * voxel_size_scale, 2)

#         num_pos = init_ply_v2(ply_path, receiver_pos, nerf_normalization["extent"], cube_size)

#         print(f"\nRandomlize point coluds. Cube size: {cube_size} meters, Number of points: {num_pos}\n")

#     try:
#         pcd = fetch_init_ply(ply_path)

#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd, 
#                            train_spectrums=train_infos,
#                            test_spectrums=test_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
    
#     return scene_info


# def fetch_init_ply(path):
    
#     plydata = PlyData.read(path)

#     vertices = plydata['vertex']

#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

#     return BasicPointCloud(points=positions, attris=None, normals=normals)


# def init_ply_v2(ply_path, receiver_pos, camera_extent, cube_size):

#     dtype = [('x', 'f4'),  ('y', 'f4'),  ('z', 'f4'),
#             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
#             ]
#     xyz = generate_cube_coordinates(receiver_pos, camera_extent, cube_size)

#     #  Initializes an array of normals with the same shape as xyz but filled with zeros
#     normals = np.zeros_like(xyz)

#     # Creates an empty structured array for the elements of the PLY file, with the specified dtype.
#     elements = np.empty(xyz.shape[0], dtype=dtype)

#     attributes = np.concatenate((xyz, normals), axis=1)

#     elements[:] = list(map(tuple, attributes))

#     vertex_element = PlyElement.describe(elements, 'vertex')

#     ply_data = PlyData([vertex_element])

#     ply_data.write(ply_path)

#     return xyz.shape[0]


# def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
#     # Define the 3D space boundaries
#     x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
#     x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()

#     y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
#     y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()
    
#     z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
#     z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

#     num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
#     num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
#     num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))

#     x_coords = np.linspace(x_min, x_max, num_cubes_x) if num_cubes_x > 1 else np.array([(x_min + x_max) / 2])
#     y_coords = np.linspace(y_min, y_max, num_cubes_y) if num_cubes_y > 1 else np.array([(y_min + y_max) / 2])
#     z_coords = np.linspace(z_min, z_max, num_cubes_z) if num_cubes_z > 1 else np.array([(z_min + z_max) / 2])

#     # Create a grid of points
#     x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
#     cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

#     return cube_points

"""
Data loader for pickle-based RF simulation data for 3D Gaussian Splatting
This function adapts the pickle data format to work with the existing SceneInfo structure
"""

import os
import sys
import pickle
import numpy as np
import torch
from typing import NamedTuple
from pathlib import Path
from plyfile import PlyData, PlyElement


class SpectrumInfo(NamedTuple):
    R: np.array           # Rotation matrix of the receiver
    T_rx: np.array        # Translation vector of the receiver
    T_tx: np.array        # Translation vector of the transmitter
    spectrum: np.array    # Spectrum/waveform data
    spectrum_path: str    # File path (dummy for pickle data)
    spectrum_name: str    # Name/index of the spectrum
    width: int            # Width of spectrum data
    height: int           # Height of spectrum data


class BasicPointCloud(NamedTuple):
    points: np.array
    attris: np.array
    normals: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_spectrums: list
    test_spectrums: list
    nerf_normalization: dict
    ply_path: str


class DummyClass:
    pass


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'data_schemas':
            return DummyClass
        return super().find_class(module, name)


def load_pickle_data(pickle_path):
    """Load data from pickle file with custom unpickler"""
    with open(pickle_path, 'rb') as f:
        data = CustomUnpickler(f).load()
    
    if hasattr(data, '_data'):
        return data._data
    return data


def process_waveform_to_spectrum(wfm_data, target_height=90, target_width=256):
    """
    Convert complex waveform data to 2D spectrum representation
    
    Args:
        wfm_data: Complex waveform data (N, time_samples)
        target_height: Desired height for spectrum image
        target_width: Desired width (should match time_samples)
    
    Returns:
        Normalized spectrum data (height, width)
    """
    # Take magnitude of complex data
    magnitude = np.abs(wfm_data)
    
    # If we need to reshape to match expected spectrum format
    if magnitude.shape[1] == target_width:
        # Already correct width, just take magnitude
        spectrum = magnitude
    else:
        # Resample if needed
        spectrum = magnitude
    
    # Normalize to [0, 1] range
    if spectrum.max() > 0:
        spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    
    return spectrum.astype(np.float32)


def getNorm_3d(tx_coords, rx_coords, scale=2.0):
    """
    Calculate normalization parameters for the scene
    
    Args:
        tx_coords: Transmitter coordinates (N, 3)
        rx_coords: Receiver coordinates (N, 3)
        scale: Scaling factor for the scene extent
    
    Returns:
        Dictionary with translation, radius, and extent information
    """
    # Use the first receiver position as the reference point (gateway)
    gateway_pos = torch.from_numpy(rx_coords[0]).float()
    
    # Convert all TX positions
    tx_positions = torch.from_numpy(tx_coords).float()
    
    # Calculate distances from gateway to all TX positions
    cam_centers = tx_positions  # Shape: (N, 3)
    gateway_pos_expanded = gateway_pos.unsqueeze(0)  # Shape: (1, 3)
    
    # Calculate distances
    dists = torch.norm(cam_centers - gateway_pos_expanded, dim=1)
    radius = torch.max(dists).item() * scale
    
    # Calculate extent in each dimension
    deviations = cam_centers - gateway_pos_expanded  # Shape: (N, 3)
    
    positive_deviations = deviations.clone()
    negative_deviations = deviations.clone()
    
    positive_deviations[positive_deviations < 0] = 0
    negative_deviations[negative_deviations > 0] = 0
    
    max_positive = positive_deviations.max(dim=0).values * scale
    max_negative = negative_deviations.min(dim=0).values.abs() * scale
    
    epsilon = 1e-6
    max_positive[max_positive < epsilon] = 1.0
    max_negative[max_negative < epsilon] = 1.0
    
    translate = -gateway_pos
    
    extent = {
        "max_positive": max_positive,
        "max_negative": max_negative
    }
    
    return {
        "translate": translate,
        "radius": radius,
        "extent": extent
    }


def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
    """Generate 3D grid of initial point cloud positions"""
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()
    
    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()
    
    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()
    
    num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
    num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
    num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))
    
    x_coords = np.linspace(x_min, x_max, num_cubes_x) if num_cubes_x > 1 else np.array([(x_min + x_max) / 2])
    y_coords = np.linspace(y_min, y_max, num_cubes_y) if num_cubes_y > 1 else np.array([(y_min + y_max) / 2])
    z_coords = np.linspace(z_min, z_max, num_cubes_z) if num_cubes_z > 1 else np.array([(z_min + z_max) / 2])
    
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
    
    return cube_points


def init_ply_from_data(ply_path, receiver_pos, camera_extent, cube_size):
    """Initialize PLY file with cube grid of points"""
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    xyz = generate_cube_coordinates(receiver_pos, camera_extent, cube_size)
    normals = np.zeros_like(xyz)
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)
    
    return xyz.shape[0]


def fetch_init_ply(path):
    """Load point cloud from PLY file"""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return BasicPointCloud(points=positions, attris=None, normals=normals)


def readPickleRFSceneInfo(pickle_path, args_model):
    """
    Read RF scene information from pickle file
    
    Args:
        pickle_path: Path to the pickle file
        args_model: Model arguments containing configuration
    
    Returns:
        SceneInfo object with all scene data
    """
    print(f"\nLoading data from pickle file: {pickle_path}")
    
    # Load pickle data
    data = load_pickle_data(pickle_path)
    
    # Extract data
    wfm_data = data['wfm_data']      # (N, time_samples) complex64
    rx_coords = data['rx_coords']    # (N, 3) float32
    tx_coords = data['tx_coords']    # (N, 3) float32
    
    print(f"Loaded {wfm_data.shape[0]} waveforms with {wfm_data.shape[1]} time samples")
    print(f"TX coords shape: {tx_coords.shape}")
    print(f"RX coords shape: {rx_coords.shape}")
    
    # Process waveforms to spectrum format
    # Assuming each waveform is a 1D time series that we treat as a "spectrum"
    spectrums_info = []
    
    # Get unique receiver position (assuming single gateway)
    unique_rx = np.unique(rx_coords, axis=0)
    if len(unique_rx) == 1:
        gateway_pos = unique_rx[0]
        # Create identity rotation for receiver
        rotation_matrix = torch.eye(3).float()
    else:
        # If multiple receivers, use the first one as reference
        gateway_pos = rx_coords[0]
        rotation_matrix = torch.eye(3).float()
    
    print(f"Gateway position: {gateway_pos}")
    
    # Create SpectrumInfo for each waveform
    for idx in range(len(wfm_data)):
        # Get waveform data for this index
        wfm = wfm_data[idx:idx+1]  # Shape: (1, time_samples)
        
        # Convert to spectrum representation (magnitude)
        spectrum = np.abs(wfm[0]).astype(np.float32)  # Shape: (time_samples,)
        
        # Normalize
        if spectrum.max() > 0:
            spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
        
        # Create 2D spectrum by treating as 1D signal
        # Height=1, Width=time_samples for now
        # Or reshape to match expected format
        height = 1
        width = spectrum.shape[0]
        
        spectrum_tensor = torch.from_numpy(spectrum).float()
        
        spec_info = SpectrumInfo(
            R=rotation_matrix,
            T_rx=torch.from_numpy(rx_coords[idx]).float(),
            T_tx=torch.from_numpy(tx_coords[idx]).float(),
            spectrum=spectrum_tensor,
            spectrum_path=f"waveform_{idx:06d}",
            spectrum_name=f"{idx:06d}",
            height=height,
            width=width
        )
        spectrums_info.append(spec_info)
    
    # Split into train/test
    eval_mode = args_model.eval
    llffhold = getattr(args_model, 'llffhold', 8)
    
    n_samples = len(spectrums_info)
    i_test = np.arange(n_samples)[::llffhold]
    i_train = np.array([j for j in np.arange(n_samples) if j not in i_test])
    
    print(f"Train samples: {len(i_train)}, Test samples: {len(i_test)}")
    
    if eval_mode:
        train_infos = [spectrums_info[i] for i in i_train]
        test_infos = [spectrums_info[i] for i in i_test]
    else:
        train_infos = spectrums_info
        test_infos = []
    
    # Calculate normalization
    camera_scale = getattr(args_model, 'camera_scale', 2.0)
    nerf_normalization = getNorm_3d(tx_coords, rx_coords, scale=camera_scale)
    
    # Initialize point cloud
    output_dir = os.path.dirname(pickle_path)
    ply_path = os.path.join(output_dir, "points3D.ply")
    
    gene_init_point = getattr(args_model, 'gene_init_point', False)
    
    if not os.path.exists(ply_path) or gene_init_point:
        # Calculate cube size based on wavelength
        # Assuming ~900 MHz frequency
        voxel_size_scale = getattr(args_model, 'voxel_size_scale', 1.0)
        frequency = 902e6  # 902 MHz
        wavelength = 3.0e8 / frequency  # Speed of light / frequency
        cube_size = round(wavelength * voxel_size_scale, 3)
        
        print(f"Initializing point cloud with cube size: {cube_size} meters")
        num_points = init_ply_from_data(
            ply_path, 
            gateway_pos, 
            nerf_normalization["extent"], 
            cube_size
        )
        print(f"Created {num_points} initial points")
    
    # Load point cloud
    try:
        pcd = fetch_init_ply(ply_path)
        print(f"Loaded point cloud with {pcd.points.shape[0]} points")
    except:
        pcd = None
        print("Warning: Could not load point cloud")
    
    # Create SceneInfo
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_spectrums=train_infos,
        test_spectrums=test_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    
    print(f"\nScene info created successfully!")
    print(f"  Point cloud: {pcd.points.shape[0] if pcd else 0} points")
    print(f"  Train spectrums: {len(train_infos)}")
    print(f"  Test spectrums: {len(test_infos)}")
    print(f"  Normalization radius: {nerf_normalization['radius']:.3f}")
    
    return scene_info


# # Example usage
# if __name__ == "__main__":
#     import argparse
    
#     # Create a simple args object for testing
#     class Args:
#         eval = True
#         llffhold = 8
#         camera_scale = 2.0
#         voxel_size_scale = 1.0
#         gene_init_point = True
    
#     args = Args()
    
#     pickle_path = "/fs/nexus-scratch/garner/SAR-GS/data/MMwave_recon/output_dir/sim_data_occ_bunny_5e09_b_0.7_2sphere.pik"
#     scene_info = readPickleRFSceneInfo(pickle_path, args)
    
#     print("\nScene loaded successfully!")


