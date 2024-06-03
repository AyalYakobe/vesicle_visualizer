import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import distance_transform_edt

scaling_factors = np.array([30, 8, 8])


def load_data(neuron_file_path):
    with h5py.File(neuron_file_path, 'r') as f:
        neuron_data = f['main'][:]
    return neuron_data


def calculate_distance_transform(neuron_data):
    return distance_transform_edt(neuron_data == 0, sampling=scaling_factors)


def create_mesh(positions):
    scaled_positions = positions.astype(np.float32) * scaling_factors
    return pv.PolyData(scaled_positions)


def visualize_perimeter(perimeter_positions):
    plotter = pv.Plotter()
    perimeter_mesh = create_mesh(perimeter_positions)
    plotter.add_mesh(perimeter_mesh, color='yellow', opacity=0.1, point_size=5, render_points_as_spheres=True)
    plotter.add_axes()

    # Set the grid spacing to 10 microns
    grid_spacing_nm = 10000  # 10 microns in nanometers
    plotter.show_bounds(grid='both', location='all', ticks='both', xlabel='X axis (nm)', ylabel='Y axis (nm)',
                        zlabel='Z axis (nm)')
    plotter.show_grid(
        xlabel='X axis (10 microns)', ylabel='Y axis (10 microns)', zlabel='Z axis (10 microns)'
    )
    plotter.show()


def load_calculate_and_visualize_perimeter(neuron_file_path, perimeter_distance_threshold_nm=1000):
    neuron_data = load_data(neuron_file_path)
    distance_transform = calculate_distance_transform(neuron_data)

    # Convert the perimeter distance threshold from nanometers to voxel distances
    perimeter_distance_threshold_voxels = (
        perimeter_distance_threshold_nm / scaling_factors[0],  # x axis
        perimeter_distance_threshold_nm / scaling_factors[1],  # y axis
        perimeter_distance_threshold_nm / scaling_factors[2]  # z axis
    )

    perimeter_mask = distance_transform <= min(perimeter_distance_threshold_voxels)
    perimeter_positions = np.argwhere(perimeter_mask)

    visualize_perimeter(perimeter_positions)

