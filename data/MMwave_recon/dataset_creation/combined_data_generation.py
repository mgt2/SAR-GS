import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import scipy.io
import argparse
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_schemas import SASDataSchema, SysParams, WfmParams
import constants as c
from geometry import create_voxels
from sas_utils import crop_wfm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and process mmWave data for different objects')
    parser.add_argument('--object_name', type=str, required=True, help='Name of the object (e.g., bunny, armadillo)')
    parser.add_argument('--f_start', type=str, required=True, help='Starting frequency (e.g., 5e9)')
    parser.add_argument('--bandwidth_factor', type=float, required=True, help='Bandwidth factor (e.g., 0.7)')
    return parser.parse_args()

def apply_occlusion_filtering_open3d(scatterer_coords, tx_coords, radius_multiplier=100):
    """
    Apply Open3D-based occlusion filtering using hidden point removal.
    """
    print(f"Applying Open3D occlusion filtering for {tx_coords.shape[0]} transmitter positions...")
    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scatterer_coords.cpu().numpy())
    
    # Calculate appropriate radius for hidden point removal
    diameter = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    radius = diameter * radius_multiplier
    
    # Initialize visibility mask
    visibility_mask = np.zeros(len(pcd.points), dtype=bool)
    
    print(f"Applying Open3D occlusion filtering for {tx_coords.shape[0]} transmitter positions...")
    
    # Process each transmitter position
    for tx_idx, tx_pos in enumerate(tqdm(tx_coords.cpu().numpy(), desc="Processing TX occlusion")):
        # Apply hidden point removal from this transmitter position
        _, visible_indices = pcd.hidden_point_removal(tx_pos.tolist(), radius)
        
        # Update visibility mask (point is visible if seen from any transmitter)
        visibility_mask[visible_indices] = True
    
    # Filter coordinates based on visibility
    visible_coords = scatterer_coords[visibility_mask]
    
    print(f"Open3D occlusion filtering: {len(visible_coords)}/{len(scatterer_coords)} points remain visible")
    
    return visible_coords, torch.from_numpy(visibility_mask)


def load_and_preprocess_object(
    object_filepath,
    desired_max_dimension=0.2,
    num_scatterers=100000,
    seed=42
):
    """
    1. Load the object mesh (PLY/OBJ/etc.),
    2. Scale to have max bounding-box dimension = desired_max_dimension,
    3. Randomly subsample to num_scatterers points.

    Returns scatterer_coords (torch.float32 on CPU initially).
    """
    np.random.seed(seed)

    # Load object using Open3D
    print(f"Loading object from {object_filepath}...")
    object_mesh = o3d.io.read_triangle_mesh(object_filepath)
    print(object_mesh)
    object_points = np.asarray(object_mesh.vertices)

    # Compute bounding box and scale
    min_xyz = object_points.min(axis=0)
    max_xyz = object_points.max(axis=0)
    current_dims = max_xyz - min_xyz
    current_max_dim = np.max(current_dims)

    scale_factor = desired_max_dimension / current_max_dim
    object_points_scaled = object_points * scale_factor

    # Center at origin
    center = (object_points_scaled.max(axis=0) + object_points_scaled.min(axis=0)) / 2
    object_points_centered = object_points_scaled - center    

    # Swap y and z coordinates
    object_points_swapped = object_points_centered.copy()
    object_points_swapped[:, 1] = object_points_centered[:, 2]
    object_points_swapped[:, 2] = object_points_centered[:, 1]    

    # Randomly subsample to num_scatterers
    n_points_total = object_points_swapped.shape[0]
    print(f"Total number of points: {n_points_total}")
    if n_points_total > num_scatterers:
        print(f"Subsampling from {n_points_total} to {num_scatterers} points")
        indices = np.random.choice(n_points_total, size=num_scatterers, replace=False)
        object_points_swapped = object_points_swapped[indices]

    # Convert to torch (CPU for now)
    scatterer_coords = torch.from_numpy(object_points_swapped).float()  # [num_scatterers, 3]

    return scatterer_coords

def compute_measurements_all_tx(
    tx_coords,             # [num_tx, 3]
    scatterer_coords,      # [num_scatterers, 3]
    k_values,              # [Nk,]   (e.g. 256)
    range_resolution_m,    # scalar
    device='cuda',
    batch_size=None,
    visibility_mask=None   # Optional visibility mask for occlusion
):
    """
    Compute the SAR measurements for all transmitters in a (num_tx, Nk) array.
    Now supports occlusion through visibility_mask.
    """
    # Move everything to device
    tx_coords = tx_coords.to(device)
    scatterer_coords = scatterer_coords.to(device)
    k_values = k_values.to(device)

    num_tx = tx_coords.shape[0]
    num_scatterers = scatterer_coords.shape[0]
    Nk = k_values.shape[0]

    # This is the output array
    sarData = torch.zeros((num_tx, Nk), dtype=torch.complex64, device=device)

    if batch_size is None:
        batch_size = 10000

    start_idx = 0
    num_batches = (num_tx + batch_size - 1) // batch_size
    
    batch_pbar = tqdm(total=num_batches, desc='Processing TX batches', position=0)
    k_pbar = tqdm(total=Nk, desc='Processing k-values', position=1, leave=False)
    
    for batch_num in range(num_batches):
        end_idx = min(start_idx + batch_size, num_tx)
        tx_batch = tx_coords[start_idx:end_idx]

        distances_batch = torch.cdist(tx_batch, scatterer_coords, p=2.0) * 2.0
        k_pbar.reset()
        
        for indK in range(Nk):
            phase = k_values[indK] * distances_batch
            temp = torch.exp(1j * phase)
            
            # Apply visibility mask if provided
            if visibility_mask is not None:
                visibility_batch = visibility_mask.unsqueeze(0).expand(tx_batch.shape[0], -1).to(device)
                temp = temp * visibility_batch.float()
            
            sarData[start_idx:end_idx, indK] = temp.sum(dim=1)
            k_pbar.update(1)

        start_idx = end_idx
        batch_pbar.update(1)

    k_pbar.close()
    batch_pbar.close()
    
    return sarData

def process_and_save_data(sarData, tx_coords, object_name, f_start, N):
    # Reshape data
    sarData = sarData.reshape(691200, N, order='F')
    
    # Set up system parameters
    sys_params = SysParams()
    sys_params[c.TX_POS] = np.array([0, -0.23, 0])
    sys_params[c.RX_POS] = np.array([0, -0.23, 0])
    sys_params[c.CENTER] = np.array([0, 0, 0])
    sys_params[c.FS] = 5000000

    wfm_params = WfmParams()

    # Create voxel geometry
    dim = 0.18
    geometry = create_voxels(-dim, dim, -dim, dim, -dim, dim, 150, 150, 150)

    # Create data schema
    airsas_data = SASDataSchema()
    airsas_data[c.WFM_DATA] = sarData
    airsas_data[c.TX_COORDS] = tx_coords
    airsas_data[c.RX_COORDS] = tx_coords
    airsas_data[c.SYS_PARAMS] = sys_params
    airsas_data[c.WFM_PARAMS] = wfm_params
    airsas_data[c.GEOMETRY] = geometry

    # Calculate waveform crop settings
    speed_of_sound = 299792458
    wfm_length = N
    K = 0.70295e14

    wfm_crop_settings = crop_wfm(
        airsas_data[c.TX_COORDS],
        airsas_data[c.RX_COORDS],
        geometry[c.CORNERS],
        wfm_length,
        airsas_data[c.SYS_PARAMS][c.FS],
        K,
        speed_of_sound
    )
    
    airsas_data[c.WFM_CROP_SETTINGS] = wfm_crop_settings

    # Format frequency in e-notation for filename
    f_start_str = f"{f_start:.0e}".replace('+', '')  # Convert to e-notation and remove '+' sign
    
    # Save the processed data in the parent directory's output_dir
    output_filename = f'sim_data_occ_{object_name}_{f_start_str}_b_{bandwidth_factor}_2sphere.pik'
    output_path = os.path.join('.', 'output_dir', output_filename)
    
    with open(output_path, 'wb') as handle:
        print(f"Saving system data to {output_path}")
        pickle.dump(airsas_data, handle)

    return airsas_data

if __name__ == "__main__":
    args = parse_args()
    object_name = args.object_name
    f_start = float(args.f_start)
    bandwidth_factor = float(args.bandwidth_factor)
    # -------------------- Input parameters --------------------
    object_filepath = f"./dataset_creation/object_pointclouds/{object_name}.obj"
    desired_max_dimension = 0.2
    num_scatterers = 125000
    seed = 42

    # Load transmitter coordinates
    tx_coords = scipy.io.loadmat('rawData_matlab/tx_coord_r230_1.mat')
    tx_coords = tx_coords['tx_coords']
    tx_coords = np.array(tx_coords)
    tx_coords[:, [1, 2]] = tx_coords[:, [2, 1]]
    tx_coords = torch.from_numpy(tx_coords).float()

    # Set up frequency parameters
    speed_of_light = 299792458
    N = 256
    B = 3599104000.0*bandwidth_factor
    freqs = torch.linspace(f_start, f_start+(B*(N-1)/N), N, dtype=torch.float64)
    k = 2*np.pi*freqs/speed_of_light
    rangeResolution_m = speed_of_light/(2*B)

    # Load and preprocess object
    scatterer_coords = load_and_preprocess_object(
        object_filepath=object_filepath,
        desired_max_dimension=desired_max_dimension,
        num_scatterers=num_scatterers,
        seed=seed
    )

    # Apply occlusion filtering if enabled
    visibility_mask = None
    # scatterer_coords_filtered, visibility_mask = apply_occlusion_filtering_open3d(
    #     scatterer_coords, tx_coords, radius_multiplier=25
    # )
    # # Compute measurements
    # visibility_mask = None
    scatterer_coords_filtered = scatterer_coords
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sarData = compute_measurements_all_tx(
        tx_coords=tx_coords,
        scatterer_coords=scatterer_coords_filtered,
        k_values=k,
        range_resolution_m=rangeResolution_m,
        device=device,
        batch_size=5000,
        visibility_mask=visibility_mask
    )

    # Visualization code (same as before)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(scatterer_coords[:, 0], scatterer_coords[:, 1], scatterer_coords[:, 2])
    ax.view_init(elev=0, azim=0)
    plt.savefig(f"./dataset_creation/object_pointclouds/{object_name}_scatterer_coords_{f_start}.png")
    ax.view_init(elev=0, azim=90)
    plt.savefig(f"./dataset_creation/object_pointclouds/{object_name}_scatterer_coords_side_{f_start}.png")    
    plt.close()    

    # Process and save the data
    sarData_cpu = sarData.cpu().numpy()
    process_and_save_data(sarData_cpu, tx_coords.numpy(), object_name, f_start, N)

    print("Done! Data has been processed and saved in output_dir")
