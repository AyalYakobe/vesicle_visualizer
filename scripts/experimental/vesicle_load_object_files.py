import numpy as np
import pyvista as pv

def display_multiple_obj_files(file_paths):
    plotter = pv.Plotter()

    # Load the meshes
    for index, file_path in enumerate(file_paths):
        mesh = pv.read(file_path)
        color = 'blue' if index == 0 else 'red' if index == 1 else 'yellow'

        # Display the original mesh
        plotter.add_mesh(mesh, color=color)

        if color == 'red':  # Assuming the second file is the red mesh
            # Compute normals for the red mesh
            mesh.compute_normals(inplace=True, point_normals=True, cell_normals=False)

            # Dilation distance calculation needs adjustment:
            # A rough approximation considering an average dilation that makes sense visually
            average_dilation_distance = np.mean([2, 2, 1.33])  # Taking average dilation
            dilated_points = mesh.points + mesh.point_normals * average_dilation_distance

            # Create a new mesh from the dilated points
            dilated_mesh = pv.PolyData(dilated_points, mesh.faces)
            plotter.add_mesh(dilated_mesh, color='green', style='wireframe', opacity=0.01)  # Adjust opacity here

    plotter.show()


def display_and_count_intersections(file_paths):
    plotter = pv.Plotter()

    # Load meshes from paths
    meshes = [pv.read(path) for path in file_paths]

    # Define a list of colors to cycle through
    color_cycle = ['blue', 'red', 'yellow']

    # Initialize the point clouds array and display them
    point_clouds = []
    for index, mesh in enumerate(meshes):
        color = color_cycle[index % len(color_cycle)]

        # Compute normals on the original mesh
        mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
        normals = mesh.point_data['Normals']

        # Create a point cloud from the mesh points
        point_cloud = pv.PolyData(mesh.points)
        point_cloud.point_data['Normals'] = normals  # Add normals to point data
        point_clouds.append(point_cloud)

        # Display the point cloud
        plotter.add_points(point_cloud, color=color)

        # If this is the red mesh, create and display the dilated mesh
        if color == 'red':
            average_dilation_distance = np.mean([2, 2, 1.33])  # Custom average dilation
            dilated_points = mesh.points + normals * average_dilation_distance
            green_mesh = pv.PolyData(dilated_points)
            plotter.add_mesh(green_mesh, color='green', style='wireframe', opacity=0.01)

    # Assuming third mesh is yellow
    yellow_mesh = point_clouds[2]

    # # Find connected components in the yellow point cloud
    # components = yellow_mesh.connectivity(largest=False)
    # labels = components.point_data['RegionId']
    # unique_labels = np.unique(labels)
    #
    # count_intersecting_features = 0
    # for label in unique_labels:
    #     component = components.threshold([label, label], scalars='RegionId')
    #     if isinstance(component, pv.PolyData) and component.n_points > 0:
    #         intersect_with_green = green_mesh.boolean_intersection(component, progress_bar=False)
    #         if intersect_with_green and intersect_with_green.n_cells > 0:
    #             count_intersecting_features += 1

    plotter.show()