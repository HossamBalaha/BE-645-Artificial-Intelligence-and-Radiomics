'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 20th, 2024
# Last Modification Date: May 21st, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import os  # For file and directory operations.
import numpy as np  # For numerical operations.
import nibabel as nib  # For handling NIfTI files.

# Define the paths to the input volume and segmentation NIfTI files.
caseVolPath = r"Data/Sample Volume.nii"  # Path to the volume NIfTI file.
caseSegPath = r"Data/Sample Segmentation.nii"  # Path to the segmentation NIfTI file.

# Define the folders to store the extracted slices.
storageVolFolder = r"Data/Volume Slices"  # Folder to store volume slices.
storageSegFolder = r"Data/Segmentation Slices"  # Folder to store segmentation slices.

# Load the NIfTI files.
caseVol = nib.load(caseVolPath)  # Load the volume NIfTI file.
caseSeg = nib.load(caseSegPath)  # Load the segmentation NIfTI file.

# Extract the data arrays from the NIfTI files.
caseVolData = caseVol.get_fdata()  # Get the volume data as a NumPy array.
caseSegData = caseSeg.get_fdata()  # Get the segmentation data as a NumPy array.

# Get the shape (dimensions) of the data arrays.
caseVolShape = caseVolData.shape  # Shape of the volume data.
caseSegShape = caseSegData.shape  # Shape of the segmentation data.

# Print the shapes of the data arrays.
print("Volume Shape: ", caseVolShape)  # Print the shape of the volume data.
print("Segmentation Shape: ", caseSegShape)  # Print the shape of the segmentation data.

# Create the storage folders if they don't exist.
if (not os.path.exists(storageVolFolder)):
  os.makedirs(storageVolFolder)  # Create the volume slices folder.
if (not os.path.exists(storageSegFolder)):
  os.makedirs(storageSegFolder)  # Create the segmentation slices folder.

# Extract and process slices from the volume and segmentation data.
for i in range(caseVolShape[2]):  # Loop through each slice along the third dimension.
  # Extract the current slice from the volume and segmentation data.
  volSlice = caseVolData[:, :, i]  # Extract the volume slice.
  segSlice = caseSegData[:, :, i]  # Extract the segmentation slice.

  # Rotate the slices 90 degrees counterclockwise for better visualization.
  volSlice = np.rot90(volSlice)  # Rotate the volume slice.
  segSlice = np.rot90(segSlice)  # Rotate the segmentation slice.

  # Modify the segmentation slice to differentiate between labels.
  # For visualization of both labels, we map label 1 to a grayscale value of 127
  # and label 2 to a grayscale value of 255.
  segSlice[segSlice == 1] = 127  # Set label 1 to a grayscale value of 127.
  segSlice[segSlice == 2] = 255  # Set label 2 to a grayscale value of 255.

  # For binary segmentation for the liver with the tumor nodes.
  # segSlice[segSlice == 1] = 255  # Set label 1 to a grayscale value of 255.
  # segSlice[segSlice == 2] = 255  # Set label 2 to a grayscale value of 255.

  # For binary segmentation for the tumor nodes only.
  # segSlice[segSlice == 1] = 0  # Set label 1 to a grayscale value of 0.
  # segSlice[segSlice == 2] = 255  # Set label 2 to a grayscale value of 255.

  # Save the slices as BMP images.
  cv2.imwrite(
    os.path.join(storageVolFolder, f"Volume Slice {i}.bmp"),  # File path for the volume slice.
    volSlice  # Volume slice data.
  )
  cv2.imwrite(
    os.path.join(storageSegFolder, f"Segmentation Slice {i}.bmp"),  # File path for the segmentation slice.
    segSlice  # Segmentation slice data.
  )
