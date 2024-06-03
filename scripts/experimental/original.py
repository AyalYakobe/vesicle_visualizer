import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops

scaling_factors = np.array([30, 8, 8])  # nm per voxel for x, y, z

def load_data(neuron_file_path, vesicle_file_path):
    with h5py.File(neuron_file_path, 'r') as f:
        neuron_data = f['main'][:]
    with h5py.File(vesicle_file_path, 'r') as f:
        vesicle_data = f['main'][:]
    return neuron_data, vesicle_data

def calculate_distance_transform(neuron_data):
    return distance_transform_edt(neuron_data == 0, sampling=scaling_factors)
# def calculate_distance_transform(neuron_data):
#     return edt.edt(neuron_data == 0, anisotropy=(30, 8, 8), black_border=True, order='F', parallel=4)

def identify_vesicles_within_perimeter(labeled_vesicles, distance_transform, perimeter_distance_threshold):
    vesicles_within_perimeter = []
    vesicles_within_perimeter_labels = set()
    for region in regionprops(labeled_vesicles):
        if np.any(distance_transform[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]] <= perimeter_distance_threshold):
            vesicles_within_perimeter.append(region.coords)
            vesicles_within_perimeter_labels.add(region.label)
    return vesicles_within_perimeter, vesicles_within_perimeter_labels

def create_mesh(positions):
    scaled_positions = positions.astype(np.float32) * scaling_factors
    return pv.PolyData(scaled_positions)

def visualize_data(neuron_mesh, vesicles_within_perimeter, vesicles_within_perimeter_labels, labeled_vesicles, perimeter_positions):
    plotter = pv.Plotter()
    plotter.add_mesh(neuron_mesh, color='blue', opacity=0.3, point_size=5, render_points_as_spheres=True)
    for vesicle in vesicles_within_perimeter:
        if len(vesicle) > 0:
            vesicle_mesh = create_mesh(vesicle)
            plotter.add_mesh(vesicle_mesh, color='green', point_size=10, render_points_as_spheres=True)
    for region in regionprops(labeled_vesicles):
        if region.label not in vesicles_within_perimeter_labels:
            vesicle_mesh = create_mesh(region.coords)
            plotter.add_mesh(vesicle_mesh, color='red', point_size=10, render_points_as_spheres=True)
    perimeter_mesh = create_mesh(perimeter_positions)
    plotter.add_mesh(perimeter_mesh, color='yellow', opacity=0.1, point_size=5, render_points_as_spheres=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

def load_calculate_and_visualize_neuron_and_vesicles(neuron_file_path, vesicle_file_path, perimeter_distance_threshold_nm=1000):
    neuron_data, vesicle_data = load_data(neuron_file_path, vesicle_file_path)

    labeled_vesicles, num_vesicles = label(vesicle_data, return_num=True)

    # Calculate the distance transform with anisotropic scaling
    distance_transform = calculate_distance_transform(neuron_data)

    # Convert the perimeter distance threshold from nanometers to voxel distances for each axis
    perimeter_distance_threshold_voxels = (
        perimeter_distance_threshold_nm / scaling_factors[0],  # x axis
        perimeter_distance_threshold_nm / scaling_factors[1],  # y axis
        perimeter_distance_threshold_nm / scaling_factors[2]   # z axis
    )

    # Use the maximum voxel distance among the axes for the threshold
    perimeter_distance_threshold_voxel = min(perimeter_distance_threshold_voxels)

    vesicles_within_perimeter, vesicles_within_perimeter_labels = identify_vesicles_within_perimeter(
        labeled_vesicles, distance_transform, perimeter_distance_threshold_voxel)

    perimeter_mask = (distance_transform <= perimeter_distance_threshold_voxel) & (neuron_data == 0)

    perimeter_positions = np.argwhere(perimeter_mask)

    neuron_positions = np.column_stack(np.nonzero(neuron_data))

    neuron_mesh = create_mesh(neuron_positions)

    visualize_data(neuron_mesh, vesicles_within_perimeter, vesicles_within_perimeter_labels, labeled_vesicles, perimeter_positions)

    num_vesicles_within_perimeter = len(vesicles_within_perimeter)
    print("Number of vesicle objects within the perimeter:", num_vesicles_within_perimeter)

