'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 6th, 2024
# Last Modification Date: Jan 28th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


def ReadVolume(caseImgPaths, caseSegPaths):
  """
  Read and preprocess a 3D volume from a set of 2D slices and their corresponding segmentation masks.

  Args:
      caseImgPaths (list): List of file paths to medical image slices in BMP format.
      caseSegPaths (list): List of file paths to segmentation masks matching the slices.

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

    # Extract region of interest using bitwise AND operation between image and mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Calculate bounding box coordinates of non-zero region in ROI.
    x, y, w, h = cv2.boundingRect(roi)
    # Crop image to tight bounding box around segmented area.
    cropped = roi[y:y + h, x:x + w]

    # Validate cropped slice contains actual data (not just background).
    if (np.sum(cropped) <= 0):
      raise ValueError("The cropped image is empty. Please check the segmentation mask.")

    # Add processed slice to volume list.
    volumeCropped.append(cropped)

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


def CalculateGLRLM3DRunLengthMatrix(volume, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate 3D Gray-Level Run-Length Matrix (GLRLM) for volumetric texture analysis.

  Parameters:
      volume (numpy.ndarray): 3D array of intensity values (z, y, x dimensions)
      theta (float): Analysis angle in radians determining 3D direction vector
      isNorm (bool): Enable matrix normalization to probability distribution
      ignoreZeros (bool): Exclude zero-valued voxels from run calculations

  Returns:
      rlMatrix (numpy.ndarray): 2D matrix of size (intensity levels × max run length)
  """
  # Calculate intensity range parameters for matrix indexing.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels
  R = np.max(volume.shape)  # Maximum possible run length

  # Initialize empty GLRLM and pixel tracking matrix.
  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(volume.shape)

  # Calculate directional components using spherical coordinates.
  dx = int(np.round(np.cos(theta) * np.sin(theta)))  # X-axis step
  dy = int(np.round(np.sin(theta) * np.sin(theta)))  # Y-axis step
  dz = int(np.round(np.cos(theta)))  # Z-axis step

  # Iterate through all voxels in 3D volume.
  for i in range(volume.shape[0]):  # Z-dimension
    for j in range(volume.shape[1]):  # Y-dimension
      for k in range(volume.shape[2]):  # X-dimension
        # Skip previously processed voxels.
        if seenMatrix[i, j, k] == 1:
          continue

        # Mark current voxel as processed.
        seenMatrix[i, j, k] = 1
        currentVal = volume[i, j, k]
        runLength = 1

        # Extend run along specified direction until value change.
        while (
          (i + runLength * dz >= 0) and
          (i + runLength * dz < volume.shape[0]) and
          (j + runLength * dy >= 0) and
          (j + runLength * dy < volume.shape[1]) and
          (k + runLength * dx >= 0) and
          (k + runLength * dx < volume.shape[2])
        ):
          if volume[i + runLength * dz, j + runLength * dy, k + runLength * dx] == currentVal:
            seenMatrix[i + runLength * dz, j + runLength * dy, k + runLength * dx] = 1
            runLength += 1
          else:
            break

        # Skip zero-value runs if configured.
        if ignoreZeros and currentVal == 0:
          continue

        # Update GLRLM with current run information.
        rlMatrix[currentVal - minA, runLength - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if isNorm:
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix


def CalculateGLRLMFeatures3D(rlMatrix, volume):
  """
  Compute texture features from 3D Gray-Level Run-Length Matrix.

  Parameters:
      rlMatrix (numpy.ndarray): Precomputed GLRLM matrix
      volume (numpy.ndarray): Original 3D volume for reference parameters

  Returns:
      dict: Dictionary containing seven standardized texture features
  """
  # Calculate intensity range parameters.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1
  R = np.max(volume.shape)

  # Compute total number of runs for normalization.
  rlN = np.sum(rlMatrix)

  # Calculate Short Run Emphasis (SRE) with inverse squared weighting.
  sre = np.sum(rlMatrix / (np.arange(1, R + 1) ** 2)).sum() / rlN

  # Calculate Long Run Emphasis (LRE) with squared run length weighting.
  lre = np.sum(rlMatrix * (np.arange(1, R + 1) ** 2)).sum() / rlN

  # Calculate Gray Level Non-Uniformity (GLN) using row sums.
  gln = np.sum(np.sum(rlMatrix, axis=1) ** 2) / rlN

  # Calculate Run Length Non-Uniformity (RLN) using column sums.
  rln = np.sum(np.sum(rlMatrix, axis=0) ** 2) / rlN

  # Calculate Run Percentage relative to total voxels.
  rp = rlN / np.prod(volume.shape)

  # Calculate Low Gray Level Emphasis (LGRE) with inverse intensity weighting.
  lgre = np.sum(rlMatrix / (np.arange(1, N + 1)[:, None] ** 2)).sum() / rlN

  # Calculate High Gray Level Emphasis (HGRE) with intensity squared weighting.
  hgre = np.sum(rlMatrix * (np.arange(1, N + 1)[:, None] ** 2)).sum() / rlN

  return {
    "Short Run Emphasis"          : sre,
    "Long Run Emphasis"           : lre,
    "Gray Level Non-Uniformity"   : gln,
    "Run Length Non-Uniformity"   : rln,
    "Run Percentage"              : rp,
    "Low Gray Level Run Emphasis" : lgre,
    "High Gray Level Run Emphasis": hgre,
  }


# Example file paths for medical imaging data.
caseImgPaths = [
  r"Data/Volume Slices/Volume Slice 65.bmp",
  r"Data/Volume Slices/Volume Slice 66.bmp",
  r"Data/Volume Slices/Volume Slice 67.bmp",
]
caseSegPaths = [
  r"Data/Segmentation Slices/Segmentation Slice 65.bmp",
  r"Data/Segmentation Slices/Segmentation Slice 66.bmp",
  r"Data/Segmentation Slices/Segmentation Slice 67.bmp",
]

# Set analysis angle to 0 degrees (horizontal direction).
theta = 0
# Convert angle to radians for trigonometric functions.
theta = np.radians(theta)

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Compute GLRLM with normalized probabilities and zero exclusion.
rlMatrix = CalculateGLRLM3DRunLengthMatrix(
  volumeCropped, theta,
  isNorm=True, ignoreZeros=True
)

# Extract texture features from computed GLRLM.
features = CalculateGLRLMFeatures3D(rlMatrix, volumeCropped)

# Display computed features with formatted output.
print(f"At angle {theta} degrees:")
for feature, value in features.items():
  # Print feature name with value rounded to 4 decimal places.
  print(f"{feature} : {np.round(value, 4)}")
