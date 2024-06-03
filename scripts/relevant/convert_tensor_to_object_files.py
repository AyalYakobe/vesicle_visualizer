import h5py
import numpy as np
import pyvista as pv
import trimesh

scaling_factors = np.array([30, 8, 8])

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = f['main'][:]
    return data

def create_mesh(positions, scaling_factors):
    scaled_positions = positions.astype(np.float32) * scaling_factors
    return pv.PolyData(scaled_positions)

def save_mesh_as_obj(mesh, file_path):
    # Extract vertices and faces from the PyVista mesh
    vertices = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]

    # Create a trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Save as .obj file
    tri_mesh.export(file_path)

def convert_h5_to_obj(neuron_file_path, vesicle_file_path, neuron_obj_path, vesicle_obj_path):
    # Load data
    neuron_data = load_data(neuron_file_path)
    vesicle_data = load_data(vesicle_file_path)

    # Get positions
    neuron_positions = np.column_stack(np.nonzero(neuron_data))
    vesicle_positions = np.column_stack(np.nonzero(vesicle_data))

    # Create meshes
    neuron_mesh = create_mesh(neuron_positions, scaling_factors)
    vesicle_mesh = create_mesh(vesicle_positions, scaling_factors)

    # Save meshes as .obj files
    save_mesh_as_obj(neuron_mesh, neuron_obj_path)
    save_mesh_as_obj(vesicle_mesh, vesicle_obj_path)
