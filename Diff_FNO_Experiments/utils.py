import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch

def read_mesh_coordinates(file_path):
    """Reads mesh cell coordinates from the specified OpenFOAM C file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Locate the start of the coordinates list
    start_index = None
    for i, line in enumerate(lines):
        if 'nonuniform List<vector>' in line:
            start_index = i + 2  # The data starts two lines after this
            break

    if start_index is None:
        raise ValueError("Start of coordinates data (nonuniform List<vector>) not found in the file.")

    coordinates = []
    for line in lines[start_index:]:
        line = line.strip()
        if line.startswith(')'):  # End of the list
            break
        if line.startswith('(') and line.endswith(')'):
            coord = line.strip('()').split()
            coordinates.append([float(c) for c in coord])

    return np.array(coordinates)

def transform_coordinates(point_coordinates, hyperparameter):
    """
    Transforms the point coordinates into a 3D array of shape [total_points, 2, hyperparameter].
    
    Args:
        point_coordinates (numpy.ndarray): Array of shape [total_points, 3] with x, y, z coordinates.
        hyperparameter (int): Number of repetitions for each coordinate value.

    Returns:
        numpy.ndarray: Transformed 3D array of shape [total_points, 2, hyperparameter].
    """
    total_points = point_coordinates.shape[0]
    # Extract x and y coordinates
    x_coords = point_coordinates[:, 0]
    y_coords = point_coordinates[:, 1]

    # Repeat each coordinate value 'hyperparameter' times
    x_repeated = np.repeat(x_coords[:, np.newaxis], hyperparameter, axis=1)  # [total_points, hyperparameter]
    y_repeated = np.repeat(y_coords[:, np.newaxis], hyperparameter, axis=1)  # [total_points, hyperparameter]

    # Stack x and y coordinates along a new axis to form a 3D array
    transformed = np.stack((x_repeated, y_repeated), axis=1)  # [total_points, 2, hyperparameter]

    return transformed

def sigma_formation(PATH_Sigma):
    sigma_data = np.load(PATH_Sigma)  # shape (N_sim=1000, T=21, N_points=16320, Channels=4)

    X_list = []  # will hold sigma(t)
    Y_list = []  # will hold sigma(t+1)
    for s in range(1000):    # loop over simulations
        for t in range(20):  # T-1 = 20
            # sigma_data[s, t, ...] is shape (16320, 4)
            # sigma_data[s, t+1, ...] is shape (16320, 4)

            X_list.append(sigma_data[s, t, :, :])      # shape (16320, 4)
            Y_list.append(sigma_data[s, t+1, :, :])    # shape (16320, 4)

    # Now stack them into arrays of shape (N_samples, 16320, 4)
    X = np.stack(X_list, axis=0)  # shape (1000*20=20000, 16320, 4)
    Y = np.stack(Y_list, axis=0)  # same shape (20000, 16320, 4)
    return X, Y