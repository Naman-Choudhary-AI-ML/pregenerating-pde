#!/usr/bin/env python3

import os
import shutil
import subprocess
import re
import random
import numpy as np

def create_design_point_folders(base_dir, num_folders, original_0_folder, original_system_folder, original_physical_properties):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Base directory created: {base_dir}")
    else:
        print(f"Base directory already exists: {base_dir}")

    # Create the design point folders and copy necessary files
    for i in range(1, num_folders + 1):
        folder_name = f"design_point_{i}"
        folder_path = os.path.join(base_dir, folder_name)
        
        # Create the design point folder
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder created: {folder_path}")
        
        # Copy the '0' folder
        shutil.copytree(original_0_folder, os.path.join(folder_path, "0"))
        print(f"Copied '0' folder to: {folder_path}/0")
        
        # Copy the 'system' folder
        shutil.copytree(original_system_folder, os.path.join(folder_path, "system"))
        print(f"Copied 'system' folder to: {folder_path}/system")
        
        # Copy the 'physicalProperties' file
        shutil.copyfile(original_physical_properties, os.path.join(folder_path, "physicalProperties"))
        print(f"Copied 'physicalProperties' file to: {folder_path}/physicalProperties")
        

def modify_geo_file(original_geo_file, modified_geo_file, new_points):
    # Read the file content from the original .geo file
    with open(original_geo_file, 'r') as file:
        lines = file.readlines()
    
    # Modify the existing points
    for i, line in enumerate(lines):
        for point, new_value in new_points.items():
            if point in line:
                lines[i] = f"{point} = {new_value};\n"
    
    # Write the modified content to the modified .geo file
    with open(modified_geo_file, 'w') as file:
        file.writelines(lines)
    
    print(f"Points replaced successfully in the modified file located at: {modified_geo_file}")

def run_gmsh(modified_geo_file):
    try:
        # Change directory to where the modified file is located
        mesh_folder_path = os.path.dirname(modified_geo_file)
        os.chdir(mesh_folder_path)
        print(f"Changed working directory to: {mesh_folder_path}")
        
        # Define the name of the mesh file
        mesh_file_name = os.path.basename(modified_geo_file).replace('.geo', '.msh')
        mesh_file_path = os.path.join(mesh_folder_path, mesh_file_name)
        
        # Run gmsh to generate the mesh
        gmsh_command = [
            "gmsh",
            os.path.basename(modified_geo_file),
            "-3",
            "-o", mesh_file_name,
            "-format", "msh2"
        ]
        subprocess.run(gmsh_command, check=True)
        print(f"3D mesh generated and saved as: {mesh_file_name}")
        
        return mesh_file_path
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Gmsh: {e}")
        return None

def run_gmshToFoam(mesh_file_path):
    try:
        mesh_folder_path = os.path.dirname(mesh_file_path)
        os.chdir(mesh_folder_path)
        # Run gmshToFoam command
        foam_command = ["gmshToFoam", os.path.basename(mesh_file_path)]
        subprocess.run(foam_command, check=True)
        print(f"Converted mesh to OpenFOAM format using gmshToFoam.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running gmshToFoam: {e}")

def run_icoFoam(mesh_file_path):
    try:
        mesh_folder_path = os.path.dirname(mesh_file_path)
        os.chdir(mesh_folder_path)
        # Run gmshToFoam command
        simulation_command = ["simpleFoam"]
        subprocess.run(simulation_command, check=True)
        print(f"Ran the Simulation.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running simulation: {e}")

def run_centercellcoordinates(mesh_file_path):
    try:
        mesh_folder_path = os.path.dirname(mesh_file_path)
        os.chdir(mesh_folder_path)
        # Run gmshToFoam command
        simulation_command = ["postProcess", "-func", "writeCellCentres"]
        subprocess.run(simulation_command, check=True)
        print(f"Ran the Simulation.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running simulation: {e}")

def run_vel_mag(mesh_file_path):
    try:
        mesh_folder_path = os.path.dirname(mesh_file_path)
        os.chdir(mesh_folder_path)
        # Run gmshToFoam command
        simulation_command = ["postProcess", "-func", "mag(U)"]
        subprocess.run(simulation_command, check=True)
        print(f"Ran the Simulation.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running simulation: {e}")

def parse_boundary_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    boundaries = {}
    num_boundaries = None
    index = 0
    
    # Skip header lines until we find the number of boundaries
    while index < len(lines):
        line = lines[index].strip()
        if line.isdigit():
            num_boundaries = int(line)
            index += 1  # Move past the number line
            break
        index += 1
    
    if num_boundaries is None:
        raise ValueError("Number of boundaries not found in the boundary file.")
    
    # Skip lines until we reach '('
    while index < len(lines) and lines[index].strip() != '(':
        index += 1
    index += 1  # Skip '(' line
    
    # Parse each boundary block
    while index < len(lines):
        line = lines[index].strip()
        if line == ')' or line.startswith('//'):
            break  # End of boundaries section
        if not line:
            index += 1
            continue
        # Get boundary name
        boundary_name = line
        index += 1  # Move to next line
        
        # Skip until '{'
        while index < len(lines) and lines[index].strip() != '{':
            index += 1
        index += 1  # Skip '{' line
        
        # Read properties
        properties = {}
        while index < len(lines):
            prop_line = lines[index].strip()
            if prop_line == '}':
                index += 1  # Move past '}'
                break
            if prop_line and not prop_line.startswith('//'):
                key_value = prop_line.rstrip(';').split()
                if len(key_value) >= 2:
                    key = key_value[0]
                    value = ' '.join(key_value[1:])
                    properties[key] = value
            index += 1
        boundaries[boundary_name] = properties
    
    return boundaries

def write_boundary_file(file_path, boundaries):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find indices for header, boundaries, and footer
    header_lines = []
    footer_lines = []
    index = 0
    num_boundaries_line_index = None
    
    # Read header
    while index < len(lines):
        line = lines[index]
        header_lines.append(line)
        if line.strip().isdigit():
            num_boundaries_line_index = index
            index += 1  # Move past the number line
            break
        index += 1
    
    # Skip lines until '('
    while index < len(lines) and lines[index].strip() != '(':
        header_lines.append(lines[index])
        index += 1
    header_lines.append(lines[index])  # Add '(' line
    index += 1  # Move past '('
    
    # Skip original boundaries
    while index < len(lines):
        if lines[index].strip() == ')':
            footer_lines = lines[index:]  # Include ')' and footer
            break
        index += 1
    
    # Reconstruct boundaries
    boundaries_lines = []
    for name, props in boundaries.items():
        boundaries_lines.append(f'    {name}\n')
        boundaries_lines.append('    {\n')
        for key, value in props.items():
            boundaries_lines.append(f'        {key}\t\t{value};\n')
        boundaries_lines.append('    }\n')
    
    # Update the number of boundaries
    num_boundaries = len(boundaries)
    header_lines[num_boundaries_line_index] = f'{num_boundaries}\n'
    
    # Combine all lines
    new_content = ''.join(header_lines)
    new_content += ''.join(boundaries_lines)
    new_content += ''.join(footer_lines)
    
    # Write back to the file
    with open(file_path, 'w') as file:
        file.write(new_content)
    
    print(f"Boundary file '{file_path}' has been updated.")


def modify_boundary_file(boundary_file_path):
    # Parse the boundary file
    boundaries = parse_boundary_file(boundary_file_path)
    
    # Modify the 'type' in the 'Cylinder' boundary
    if 'Cylinder' in boundaries:
        boundaries['Cylinder']['type'] = 'wall'
    else:
        print("Cylinder boundary not found.")

    if 'FrontBack' in boundaries:
        boundaries['FrontBack']['type'] = 'empty'
    else:
        print("FrontBack boundaries not found")
    
    if 'Inlet' in boundaries:
        boundaries['Inlet']['type'] = 'patch'
    else:
        print("Inlet boundary not found")
    
    if 'Outlet' in boundaries:
        boundaries['Outlet']['type'] = 'patch'
    else:
        print("Outlet boundary not found")
    
    if 'Top' in boundaries:
        boundaries['Top']['type'] = 'wall'
    else:
        print("Top boundary not found")
    
    if 'Bottom' in boundaries:
        boundaries['Bottom']['type'] = 'wall'
    else:
        print("Bottom boundary not found")
    
    for props in boundaries.values():
        props.pop('physicalType', None)  # Remove 'physicalType' if it exists

    
    # Write back the modified boundaries to the file
    write_boundary_file(boundary_file_path, boundaries)

def extract_coordinates(c_file_path):
    x_coords = []
    y_coords = []
    with open(c_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line with the number of data points
    num_points = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.isdigit():
            num_points = int(line)
            index += 1  # Move past the number line
            break
        index += 1
    
    if num_points is None:
        raise ValueError(f"Number of data points not found in file {c_file_path}.")
    
    # Find the opening parenthesis '('
    while index < len(lines) and lines[index].strip() != '(':
        index += 1
    index += 1  # Move past '('
    
    # Extract coordinates
    for _ in range(num_points):
        if index >= len(lines):
            break  # Avoid index out of range
        line = lines[index].strip()
        if line == ')':
            break
        # Remove any surrounding parentheses
        coord_str = line.strip('()')
        # Split into components
        coords = coord_str.split()
        if len(coords) >= 2:
            x_coords.append(float(coords[0]))
            y_coords.append(float(coords[1]))
        else:
            print(f"Unexpected coordinate format in file {c_file_path}: {line}")
        index += 1
    
    return np.array(x_coords), np.array(y_coords)

def extract_magnitude_u(mag_u_file_path):
    mag_u_values = []
    
    with open(mag_u_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the line with the number of data points
    num_points = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.isdigit():
            num_points = int(line)
            index += 1  # Move past the number line
            break
        index += 1
    
    if num_points is None:
        raise ValueError(f"Number of data points not found in file {mag_u_file_path}.")
    
    # Find the opening parenthesis '('
    while index < len(lines) and lines[index].strip() != '(':
        index += 1
    index += 1  # Move past '('
    
    # Extract magnitude values
    for _ in range(num_points):
        if index >= len(lines):
            break  # Avoid index out of range
        line = lines[index].strip()
        if line == ')':
            break
        mag_u_values.append(float(line))
        index += 1
    
    return np.array(mag_u_values)

def main():
    # Define the original and backup file paths
    original_geo_file = os.path.expanduser("~/Desktop/OpenFoam_demo/Design_Point_0/main.geo")
    original_0_folder = os.path.expanduser("~/Desktop/OpenFoam_demo/Design_Point_0/0")
    original_system_folder = os.path.expanduser("~/Desktop/OpenFoam_demo/Design_Point_0/system")
    original_physical_properties = os.path.expanduser("~/Desktop/OpenFoam_demo/Design_Point_0/physicalProperties")
    original_momentumTransport = os.path.expanduser("~/Desktop/OpenFoam_demo/Design_Point_0/constant/momentumTransport")
    
    # Define the base directory for design points
    base_dir = os.path.expanduser("~/Desktop/OpenFoam_demo/design_points")
    
    # Define the number of design point folders to create
    num_folders = 2  # Replace with your desired number
    
    # Create design point folders and copy necessary files
    create_design_point_folders(base_dir, num_folders, original_0_folder, original_system_folder, original_physical_properties)
    
    all_x_coords = []
    all_y_coords = []
    all_mag_u = []

    # For each design point, modify the geometry, generate the mesh, run gmshToFoam, and modify the boundary file
    for i in range(1, num_folders + 1):
        folder_name = f"design_point_{i}"
        folder_path = os.path.join(base_dir, folder_name)
        
        # Define the path for the modified .geo file in this design point folder
        modified_geo_file = os.path.join(folder_path, "untitled_modified.geo")
        
        # Copy the original .geo file to the design point folder
        shutil.copyfile(original_geo_file, modified_geo_file)
        print(f"Copied original .geo file to: {modified_geo_file}")
        
        x13 = random.uniform(-9, 9)  # Ensures x6 and x7 stay within 1 to 29
        y13 = random.uniform(-6.5, 6.5)  # Ensures y8 and y9 stay within 1 to 14
        
        # Compute other points based on Point(5)
        x2 = x13 - 5

        x3 = x13 + 5

        x6 = x2

        x7 = x3
        
        x9 = x13 - 0.35355339
        y9 = y13 - 0.35355339 
        
        x10 = x13 + 0.35355339
        y10 = y13 - 0.35355339
        
        x11 = x13 - 0.35355339
        y11 = y13 + 0.35355339
        
        x12 = x13 + 0.35355339
        y12 = y13 + 0.35355339
        
        # Define the new points
        new_points = {
            "Point(2)": f"{{{x2}, -7.5, 0, 1.0}}",
            "Point(3)": f"{{{x3}, -7.5, 0, 1.0}}",
            "Point(6)": f"{{{x6}, 7.5, 0, 1.0}}",
            "Point(7)": f"{{{x7}, 7.5, 0, 1.0}}",
            "Point(9)": f"{{{x9}, {y9}, 0, 1.0}}",
            "Point(10)": f"{{{x10}, {y10}, 0, 1.0}}",
            "Point(11)": f"{{{x11}, {y11}, 0, 1.0}}",
            "Point(12)": f"{{{x12}, {y12}, 0, 1.0}}",
            "Point(13)": f"{{{x13}, {y13}, 0, 1.0}}"
        }
        
        # Modify the .geo file with the new points
        modify_geo_file(modified_geo_file, modified_geo_file, new_points)
        
        # Generate the mesh using gmsh
        mesh_file_path = run_gmsh(modified_geo_file)
        
        if mesh_file_path:
            # Run gmshToFoam to convert the mesh to OpenFOAM format
            run_gmshToFoam(mesh_file_path)
            
            # The boundary file is generated in constant/polyMesh/boundary
            # Modify the boundary file
            constant_dir = os.path.join(folder_path, "constant")
            polyMesh_dir = os.path.join(constant_dir, "polyMesh")
            boundary_file_path = os.path.join(polyMesh_dir, "boundary")
            
            if os.path.exists(boundary_file_path):
                modify_boundary_file(boundary_file_path)
            else:
                print(f"Boundary file not found: {boundary_file_path}")
            
            # Copy physicalProperties to the constant directory
            shutil.copyfile(original_physical_properties, os.path.join(constant_dir, "physicalProperties"))
            print(f"Copied 'physicalProperties' file to: {constant_dir}/physicalProperties")

            shutil.copyfile(original_momentumTransport, os.path.join(constant_dir, "momentumTransport"))
            print(f"Copied 'physicalProperties' file to: {constant_dir}/physicalProperties")

            run_icoFoam(mesh_file_path)
            run_centercellcoordinates(mesh_file_path)
            run_vel_mag(mesh_file_path)
            # run_forced(mesh_file_path)
            # Extract coordinates from 'C' file
            c_file_path = os.path.join(folder_path, "10", "C")
            if os.path.exists(c_file_path):
                x_coords, y_coords = extract_coordinates(c_file_path)
                all_x_coords.append(x_coords)
                all_y_coords.append(y_coords)
                print(f"Extracted coordinates from {c_file_path}")
            else:
                print(f"'C' file not found in {c_file_path}")
            
            # Extract magnitude U from the 'mag(U)' file
            mag_u_file_path = os.path.join(folder_path, "10", "mag(U)")
            if os.path.exists(mag_u_file_path):
                mag_u_values = extract_magnitude_u(mag_u_file_path)
                all_mag_u.append(mag_u_values)
                print(f"Extracted magnitude U from {mag_u_file_path}")
            else:
                print(f"'mag(U)' file not found in {mag_u_file_path}")
        else:
            print(f"Mesh generation failed for design point {i}")
    # Combine all x and y coordinates into 3D numpy arrays
    x_coords_np = np.array(all_x_coords)
    y_coords_np = np.array(all_y_coords)

    # Create a 3D array where the first 2D array is X coordinates and the second 2D array is Y coordinates
    coordinates_np = np.stack([x_coords_np, y_coords_np], axis=0)

    # Save the 3D array to an .npy file
    output_npy = os.path.join(base_dir, "all_coordinates.npy")
    np.save(output_npy, coordinates_np)
    print(f"All coordinates saved to {output_npy} in .npy format")

    # Combine all magnitude U values into a 2D numpy array
    mag_u_np = np.array(all_mag_u)

    # Save the 2D array of magnitude U to an .npy file
    output_mag_u_npy = os.path.join(base_dir, "all_mag_u.npy")
    np.save(output_mag_u_npy, mag_u_np)
    print(f"All magnitude U values saved to {output_mag_u_npy} in .npy format")
if __name__ == "__main__":
    main()
