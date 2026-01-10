'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Feb 25th, 2025
# Last Modification Date: Jul 7th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import trimesh  # For 3D mesh operations and visualization.
import os  # For file path operations.
from HMB_Summer_2025_Helpers import *

# Example file paths for medical imaging data.
caseImgPaths = [
  rf"Data/Volume Slices/{f}"
  for f in os.listdir(r"Data/Volume Slices")
]
caseSegPaths = [
  rf"Data/Segmentation Slices/{f}"
  for f in os.listdir(r"Data/Segmentation Slices")
]

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(
  caseImgPaths,  # Paths to image slices.
  caseSegPaths,  # Paths to segmentation slices.
  raiseErrors=False,  # Do not raise errors.
)

# Create a 3D mesh from the preprocessed volume data.
mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volumeCropped)

# Create a scene object to hold the mesh.
scene = mesh.scene()

# Visualize the 3D mesh using trimesh.
scene.show(resolution=(500, 500))
