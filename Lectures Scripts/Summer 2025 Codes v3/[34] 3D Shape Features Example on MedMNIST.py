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
import numpy as np  # For numerical operations.
from medmnist import AdrenalMNIST3D  # For loading the 3D dataset.
from HMB_Summer_2025_Helpers import *

valDataset = AdrenalMNIST3D(split="val", download=True)

print("Number of samples in the validation set:", len(valDataset))
print("Sample shape:", valDataset[0][0][0].shape)
print("Maximum value in the sample:", np.max(valDataset[0][0][0]))
print("Minimum value in the sample:", np.min(valDataset[0][0][0]))

# Load the first sample from the validation dataset.
sampleVolume = valDataset[0][0][0]
minValue = np.min(sampleVolume)
maxValue = np.max(sampleVolume)
sampleVolume = (sampleVolume - minValue) / (maxValue - minValue)  # Normalize the volume.
sampleVolume = sampleVolume.astype(np.uint8)  # Convert to uint8 type.

# Extract shape features from the preprocessed volume.
shapeFeatures = ShapeFeatures3D(sampleVolume)

# Print the shape features.
# Print the calculated shape features.
print("Shape Features:")
for feature, value in shapeFeatures.items():
  print(f"{feature}: {value:0.4f}")

# Create a 3D mesh from the preprocessed volume data.
mesh = trimesh.voxel.ops.matrix_to_marching_cubes(sampleVolume)

# Set a uniform color for the mesh (e.g., light gray).
mesh.visual.face_colors = [192, 192, 192, 255]  # RGBA: Light gray with full opacity

# Enable smooth shading for better visualization.
mesh = mesh.smoothed()

# Create a scene object to hold the mesh.
scene = mesh.scene()

# Add directional lighting to the scene.
scene.camera_transform = scene.camera.look_at(
  points=mesh.vertices,
  center=mesh.centroid,
  distance=mesh.extents.max() * 2  # Camera distance based on mesh size
)

# Visualize the 3D mesh using trimesh.
scene.show(resolution=(500, 500))
