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
      caseImgPaths (list): List of paths to the 2D slices of the volume.
      caseSegPaths (list): List of paths to the segmentation masks of the slices.

  Returns:
      volumeCropped (numpy.ndarray): A 3D NumPy array representing the preprocessed volume.
  """
  volumeCropped = []  # Initialize a list to store the cropped slices.

  # Loop through each slice and its corresponding segmentation mask.
  for i in range(len(caseImgPaths)):
    # Check if the files exist.
    if (not os.path.exists(caseImgPaths[i])) or (not os.path.exists(caseSegPaths[i])):
      raise FileNotFoundError("One or more files were not found. Please check the file paths.")

    # Load the slice and segmentation mask in grayscale mode.
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the slice.
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

    # Extract the Region of Interest (ROI) using the segmentation mask.
    roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

    # Crop the ROI to remove unnecessary background.
    x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
    cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

    if (np.sum(cropped) <= 0):
      raise ValueError("The cropped image is empty. Please check the segmentation mask.")

    # Append the cropped slice to the list.
    volumeCropped.append(cropped)

  # Determine the maximum width and height across all cropped slices.
  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])  # Maximum width.
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])  # Maximum height.

  # Pad each cropped slice to match the maximum width and height.
  for i in range(len(volumeCropped)):
    # Calculate the padding size.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]  # Horizontal padding.
    deltaHeight = maxHeight - volumeCropped[i].shape[0]  # Vertical padding.

    # Add padding to the cropped image and place the image in the center.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],  # Image to pad.
      deltaHeight // 2,  # Top padding.
      deltaHeight - deltaHeight // 2,  # Bottom padding.
      deltaWidth // 2,  # Left padding.
      deltaWidth - deltaWidth // 2,  # Right padding.
      cv2.BORDER_CONSTANT,  # Padding type.
      value=0  # Padding value.
    )

    # Replace the cropped slice with the padded slice.
    volumeCropped[i] = padded.copy()

  # Convert the list of slices to a 3D NumPy array.
  volumeCropped = np.array(volumeCropped)

  return volumeCropped  # Return the preprocessed 3D volume.


def CalculateGLRLM3DRunLengthMatrix(volume, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate the 3D Gray-Level Run-Length Matrix (GLRLM) for a given volume.

  The GLRLM is a texture analysis tool that quantifies the distribution of run lengths
  (consecutive pixels with the same intensity) in a specific direction within a 3D volume.

  Parameters:
  -----------
  volume : numpy.ndarray
      A 3D numpy array representing the volume (e.g., a medical image or a 3D texture).

  theta : float
      The angle (in radians) specifying the direction in which the run lengths are calculated.
      The direction is determined by the vector (dx, dy, dz) derived from the angle.

  isNorm : bool, optional (default=True)
      If True, the run-length matrix is normalized by dividing each entry by the sum of all entries.
      Normalization ensures that the matrix represents a probability distribution.

  ignoreZeros : bool, optional (default=True)
      If True, pixels with an intensity value of 0 are ignored in the run-length calculation.
      This is useful for excluding background pixels in medical images.

  Returns:
  --------
  rlMatrix : numpy.ndarray
      A 2D numpy array representing the run-length matrix. The rows correspond to intensity levels,
      and the columns correspond to run lengths. The value at (i, j) represents the number of runs
      of intensity level i with run length j.

  Notes:
  ------
  - The run-length matrix is calculated by traversing the volume in the specified direction (theta).
  - The matrix is initialized with dimensions (N, R), where N is the number of unique intensity levels
    and R is the maximum possible run length in the volume.
  - The matrix is updated by counting runs of consecutive pixels with the same intensity value.
  - If `ignoreZeros` is True, runs of intensity 0 are not counted.
  - If `isNorm` is True, the matrix is normalized to represent a probability distribution.
  """

  # Determine the number of unique intensity levels in the volume.
  minA = np.min(volume)  # Minimum intensity value.
  maxA = np.max(volume)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  R = np.max(volume.shape)  # Maximum run length.

  rlMatrix = np.zeros((N, R))  # Initialize the run-length matrix.
  seenMatrix = np.zeros(volume.shape)  # Initialize a matrix to track seen pixels.
  dx = int(np.round(np.cos(theta) * np.sin(theta)))  # X-direction step.
  dy = int(np.round(np.sin(theta) * np.sin(theta)))  # Y-direction step.
  dz = int(np.round(np.cos(theta)))  # Z-direction step.

  for i in range(volume.shape[0]):  # Z-axis.
    for j in range(volume.shape[1]):  # Y-axis.
      for k in range(volume.shape[2]):  # X-axis.
        # Skip if already seen.
        if (seenMatrix[i, j, k] == 1):
          continue

        seenMatrix[i, j, k] = 1  # Mark as seen.
        currentPixel = volume[i, j, k]  # Current pixel value.
        d = 1  # Distance.

        while (
          (i + d * dz >= 0) and
          (i + d * dz < volume.shape[0]) and
          (j + d * dy >= 0) and
          (j + d * dy < volume.shape[1]) and
          (k + d * dx >= 0) and
          (k + d * dx < volume.shape[2])
        ):
          if (volume[i + d * dz, j + d * dy, k + d * dx] == currentPixel):
            seenMatrix[int(i + d * dz), int(j + d * dy), int(k + d * dx)] = 1
            d += 1
          else:
            break

        # Ignore zeros if needed.
        if (ignoreZeros and (currentPixel == 0)):
          continue

        # Update the run-length matrix.
        # (- minA) is added to work with matrices that does not start from 0.
        rlMatrix[currentPixel - minA, d - 1] += 1

  if (isNorm):
    # Normalize the run-length matrix.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix  # Return the run-length matrix.


def CalculateGLRLMFeatures3D(rlMatrix, volume):
  """
  Calculate Gray Level Run Length Matrix (GLRLM) features for a 3D volume.

  This function computes various texture features based on the Gray Level Run Length Matrix (GLRLM)
  for a given 3D volume. These features are commonly used in texture analysis and can provide
  information about the distribution of gray levels and run lengths in the volume.

  Parameters:
  -----------
  rlMatrix : numpy.ndarray
      The run-length matrix computed from the 3D volume. This matrix represents the frequency
      of runs of consecutive pixels with the same intensity level.

  volume : numpy.ndarray
      The 3D volume from which the GLRLM features are to be calculated. This is typically a
      medical image volume (e.g., MRI, CT).

  Returns:
  --------
  dict
      A dictionary containing the following GLRLM-based texture features:
      - "Short Run Emphasis" (SRE): Measures the distribution of short runs in the volume.
      - "Long Run Emphasis" (LRE): Measures the distribution of long runs in the volume.
      - "Gray Level Non-Uniformity" (GLN): Measures the variability of gray levels in the volume.
      - "Run Length Non-Uniformity" (RLN): Measures the variability of run lengths in the volume.
      - "Run Percentage" (RP): Measures the fraction of the volume that is occupied by runs.
      - "Low Gray Level Run Emphasis" (LGRE): Measures the distribution of low gray level runs.
      - "High Gray Level Run Emphasis" (HGRE): Measures the distribution of high gray level runs.

  Notes:
  ------
  - The run-length matrix (rlMatrix) should be precomputed before passing it to this function.
  - The volume is used to determine the number of unique intensity levels and the maximum run length.
  - The features are normalized by the total number of runs (rlN) to ensure they are scale-invariant.
  """

  # Determine the number of unique intensity levels in the volume.
  minA = np.min(volume)  # Minimum intensity value.
  maxA = np.max(volume)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  R = np.max(volume.shape)  # Maximum run length.

  rlN = np.sum(rlMatrix)  # Total number of runs.

  # Short Run Emphasis (SRE): Measures the distribution of short runs.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Long Run Emphasis (LRE): Measures the distribution of long runs.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Gray Level Non-Uniformity (GLN): Measures the variability of gray levels.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Sum of each row.
  ) / rlN

  # Run Length Non-Uniformity (RLN): Measures the variability of run lengths.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Sum of each column.
  ) / rlN

  # Run Percentage (RP): Measures the fraction of the volume occupied by runs.
  rp = rlN / np.prod(volume.shape)

  # Low Gray Level Run Emphasis (LGRE): Measures the distribution of low gray level runs.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # High Gray Level Run Emphasis (HGRE): Measures the distribution of high gray level runs.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  return {
    "Short Run Emphasis"          : sre,
    "Long Run Emphasis"           : lre,
    "Gray Level Non-Uniformity"   : gln,
    "Run Length Non-Uniformity"   : rln,
    "Run Percentage"              : rp,
    "Low Gray Level Run Emphasis" : lgre,
    "High Gray Level Run Emphasis": hgre,
  }


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

theta = 0  # Angle in degrees.
theta = np.radians(theta)  # Convert angle to radians.

volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)  # Read and preprocess the volume.

rlMatrix = CalculateGLRLM3DRunLengthMatrix(
  volumeCropped, theta,
  isNorm=True, ignoreZeros=True
)  # Calculate the run-length matrix.

features = CalculateGLRLMFeatures3D(rlMatrix, volumeCropped)  # Calculate GLRLM features.

# Print the GLRLM features.
print(f"At angle {theta} degrees:")
for key in features:
  print(key, ":", np.round(features[key], 4))  # Print each feature rounded to 4 decimal places.
