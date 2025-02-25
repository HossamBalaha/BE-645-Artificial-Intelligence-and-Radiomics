'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Feb 25th, 2025
# Last Modification Date: Feb 25th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import trimesh
import os
import cv2
import numpy as np


def ReadVolumeAdcanced(caseImgPaths, caseSegPaths, specificClasses=[]):
  """
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Args:
      caseImgPaths (list): List of file paths to medical image slices in BMP format.
      caseSegPaths (list): List of file paths to segmentation masks matching the slices.
      specificClasses (list): List of specific classes to include in the segmentation.
        If empty, all classes are included.

  Returns:
      volumeCropped (numpy.ndarray): 3D array of preprocessed and aligned medical imaging data.
  """
  # Initialize empty list to store processed slices.
  volumeCropped = []

  # Process each image-segmentation pair in the input lists.
  for i in range(len(caseImgPaths)):
    # Verify both image and segmentation files exist before processing.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load grayscale medical image slice (8-bit depth).
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)
    # Load corresponding binary segmentation mask.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)

    # Check if specific classes are provided for segmentation.
    if (specificClasses):
      # Create a mask for the specific classes.
      mask = np.zeros_like(caseSeg)
      for classId in specificClasses:
        mask[caseSeg == classId] = 255
      caseSeg = mask

    # Extract region of interest using bitwise AND operation between image and mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Calculate bounding box coordinates of non-zero region in ROI.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop image to tight bounding box around segmented area.
    cropped = roi[y:y + h, x:x + w]

    # Validate cropped slice contains actual data (not just background).
    if (np.sum(cropped) <= 0):
      continue  # Skip empty slices.

    # Add processed slice to volume list.
    volumeCropped.append(cropped)

  # Check if any slices were successfully processed.
  if (len(volumeCropped) == 0):
    raise ValueError("No slices were successfully processed. Please check the input data.")

  # Determine maximum dimensions across all slices for padding alignment.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])

  # Standardize slice dimensions through symmetric padding.
  for i in range(len(volumeCropped)):
    # Calculate required padding for width and height dimensions.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]
    deltaHeight = maxHeight - volumeCropped[i].shape[0]

    # Apply padding to create uniform slice dimensions.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],
      deltaHeight // 2,  # Top padding (integer division)
      deltaHeight - deltaHeight // 2,  # Bottom padding (remainder)
      deltaWidth // 2,  # Left padding
      deltaWidth - deltaWidth // 2,  # Right padding
      cv2.BORDER_CONSTANT,  # Padding style (constant zero values)
      value=0
    )

    # Update volume with padded slice.
    volumeCropped[i] = padded.copy()

  # Convert list of 2D slices into 3D numpy array (z, y, x).
  volumeCropped = np.array(volumeCropped)

  return volumeCropped


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
specificClasses = []  # Example specific classes to include.
volumeCropped = ReadVolumeAdcanced(
  caseImgPaths, caseSegPaths,
  specificClasses=specificClasses
)

# Create a 3D mesh from the preprocessed volume data.
mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volumeCropped)

# Create a scene object to hold the mesh.
scene = mesh.scene()

# Visualize the 3D mesh using trimesh.
scene.show(resolution=(500, 500))
