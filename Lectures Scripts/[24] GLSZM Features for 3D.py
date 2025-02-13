'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 13th, 2024
# Last Modification Date: Feb 13th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import sys  # For system-specific parameters and functions.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 6)


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


def FindConnected3DRegions(volume, connectivity=6):
  """
  Finds connected regions in a 3D volume based on pixel connectivity.
  Parameters:
      volume (numpy.ndarray): A 3D NumPy array representing the input volume.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 6: 6-connectivity (faces only).
                          - 26: 26-connectivity (faces, edges, and corners).
  Returns:
      dict: A dictionary where keys are unique pixel values from the volume,
            and values are lists of sets. Each set contains the coordinates
            (i, j, k) of pixels belonging to a connected region for that pixel value.
  """

  def RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity=6):
    """
    Recursive helper function to find all connected pixels for a given starting pixel.
    Parameters:
        i (int): Z-axis index of the current pixel.
        j (int): Y-axis index of the current pixel.
        k (int): X-axis index of the current pixel.
        currentPixel (int): The pixel value being processed.
        region (set): A set to store the coordinates of connected pixels.
        seenMatrix (numpy.ndarray): A 3D matrix to track visited pixels.
        connectivity (int): The type of connectivity (6 or 26).
    Returns:
        None: The function modifies the `region` and `seenMatrix` in place.
    """
    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if Z-axis index is out of bounds.
      (i >= volume.shape[0]) or
      (j < 0) or  # Check if Y-axis index is out of bounds.
      (j >= volume.shape[1]) or
      (k < 0) or  # Check if X-axis index is out of bounds.
      (k >= volume.shape[2]) or
      (volume[i, j, k] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j, k) in region)  # Check if the pixel has already been added to the region.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j, k))  # Add the pixel coordinates to the region set.
    seenMatrix[i, j, k] = 1  # Mark the pixel as seen.

    # Recursively check the neighboring pixels (faces only for 6-connectivity).
    RecursiveHelper(i - 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor below.
    RecursiveHelper(i, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor left.
    RecursiveHelper(i, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor behind.
    RecursiveHelper(i + 1, j, k, currentPixel, region, seenMatrix, connectivity)  # Check Z-axis neighbor above.
    RecursiveHelper(i, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Check Y-axis neighbor right.
    RecursiveHelper(i, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Check X-axis neighbor front.

    # If 26-connectivity is specified, also check diagonal neighbors (edges and corners).
    if (connectivity == 26):
      RecursiveHelper(i - 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i - 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i + 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.
      RecursiveHelper(i, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)  # Diagonal neighbor.

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(volume.shape)  # Create a 3D matrix of zeros.

  # Dictionary to store regions grouped by pixel values.
  regions = {}  # Keys are pixel values, values are lists of sets.

  # Iterate over each voxel in the volume.
  for i in range(volume.shape[0]):  # Loop over Z-axis.
    for j in range(volume.shape[1]):  # Loop over Y-axis.
      for k in range(volume.shape[2]):  # Loop over X-axis.
        # Skip if the voxel has already been processed.
        if (seenMatrix[i, j, k]):
          continue  # Skip already processed voxels.

        # Get the current voxel value.
        currentPixel = volume[i, j, k]  # Retrieve the intensity value of the voxel.

        # Initialize a list for this pixel value if it doesn't exist.
        if (currentPixel not in regions):
          regions[currentPixel] = []  # Create a new list for this intensity value.

        # Initialize a new region set for the current voxel.
        region = set()  # Create an empty set to store connected voxel coordinates.

        # Use the helper function to find all connected voxels.
        RecursiveHelper(i, j, k, currentPixel, region, seenMatrix, connectivity)  # Find connected region.

        # Add the region to the dictionary if it contains any voxels.
        if (len(region) > 0):
          regions[currentPixel].append(region)  # Append the region to the list for this intensity value.

  # Return the dictionary of regions.
  return regions  # Return the dictionary containing connected regions.


def CalculateGLSZM3DSizeZoneMatrix(volume, connectivity=6, isNorm=True, ignoreZeros=True):
  """
  Calculate the Size-Zone Matrix for a 3D volume based on connected regions.
  Parameters:
      volume (numpy.ndarray): A 3D NumPy array representing the input volume.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 6: 6-connectivity (faces only).
                          - 26: 26-connectivity (faces, edges, and corners).
      isNorm (bool): Whether to normalize the size-zone matrix.
      ignoreZeros (bool): Whether to ignore zero pixel values.
  Returns:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      szDict (dict): A dictionary where keys are unique pixel values from the volume,
                     and values are lists of sets. Each set contains the coordinates
                     (i, j, k) of pixels belonging to a connected region for that pixel value.
      N (int): The number of unique pixel values in the volume.
      Z (int): The maximum size of any region in the dictionary.
  """

  if (volume.ndim != 3):
    raise ValueError("The input volume must be a 3D array.")

  if (connectivity not in [6, 26]):
    raise ValueError("Connectivity must be either 6 or 26.")

  if (volume.size == 0):
    raise ValueError("The input volume is empty.")

  if (np.max(volume) == 0):
    raise ValueError("The input volume is completely black.")

  # Find connected regions in the volume.
  szDict = FindConnected3DRegions(volume, connectivity=connectivity)  # Identify connected regions.

  # Determine the number of unique pixel values in the volume.
  minA = np.min(volume)
  maxA = np.max(volume)
  N = maxA - minA + 1  # Number of discrete intensity levels

  # Find the maximum size of any region in the dictionary.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])  # Find the largest connected region size.

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))  # Create a 2D matrix to store size-zone counts.

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentVal, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentVal == 0)):
        continue  # Skip zero-valued regions if ignoreZeros is True.

      # Increment the count for the corresponding pixel value and region size.
      szMatrix[currentVal - minA, len(zone) - 1] += 1  # Update the size-zone matrix.

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize by total sum to avoid division by zero.
    szMatrix = szMatrix / (np.sum(szMatrix) + 1e-6)

    # Return the size-zone matrix, dictionary, and metadata.
  return szMatrix, szDict, N, Z  # Return the computed outputs.


def CalculateGLSZMFeatures(szMatrix, data, N, Z):
  """
  Calculate the features of the Size-Zone Matrix (GLSZM).

  Parameters:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      N (int): The number of unique pixel values in the image.
      Z (int): The maximum size of any region in the dictionary.

  Returns:
      dict: A dictionary containing the calculated features.
  """
  # Calculate the total number of zones in the size-zone matrix.
  # Sum all values in the size-zone matrix to get the total zone count.
  szN = np.sum(szMatrix)

  # Small Zone Emphasis.
  sze = np.sum(
    szMatrix / ((np.arange(1, Z + 1) ** 2) + 1e-10),  # Divide each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Emphasis.
  lze = np.sum(
    szMatrix * ((np.arange(1, Z + 1) ** 2) + 1e-10),  # Multiply each zone by its size squared.
  ).sum() / szN  # Normalize by the total number of zones.

  # Gray Level Non-Uniformity.
  gln = np.sum(
    np.sum(szMatrix, axis=1) ** 2,  # Sum each row and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Non-Uniformity.
  zsn = np.sum(
    np.sum(szMatrix, axis=0) ** 2,  # Sum each column and square the result.
  ) / szN  # Normalize by the total number of zones.

  # Zone Percentage.
  # Divide the total number of zones by the total number of pixels.
  zp = szN / np.prod(data.shape)

  # Gray Level Variance.
  glv = np.sum(
    # Compute variance for each gray level.
    (np.sum(szMatrix, axis=1)) *
    ((np.arange(1, N + 1) - np.mean(np.arange(1, N + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Variance.
  zsv = np.sum(
    # Compute variance for zone sizes.
    (np.sum(szMatrix, axis=0)) *
    ((np.arange(1, Z + 1) - np.mean(np.arange(1, Z + 1))) ** 2),
  ) / szN  # Normalize by the total number of zones.

  # Zone Size Entropy.
  log = np.log2(szMatrix + 1e-10)  # Compute log base 2 of the size-zone matrix.
  log[log == -np.inf] = 0  # Replace -inf with 0.
  log[log < 0] = 0  # Replace negative values with 0.
  zse = np.sum(
    # Compute entropy for zone sizes.
    szMatrix * log,
  ) / szN  # Normalize by the total number of zones.

  # Low Gray Level Zone Emphasis.
  lgze = np.sum(
    # Divide each gray level by its squared value.
    szMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # High Gray Level Zone Emphasis.
  hgze = np.sum(
    # Multiply each gray level by its squared value.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone Low Gray Level Emphasis.
  # Adding 1e-10 to avoid division by zero.
  szlge = np.sum(
    # Combine small zone and low gray level emphasis.
    szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + 1e-10),
  ).sum() / szN  # Normalize by the total number of zones.

  # Small Zone High Gray Level Emphasis.
  szhge = np.sum(
    # Combine small zone and high gray level emphasis.
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / ((np.arange(1, Z + 1) ** 2) + 1e-10),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone Low Gray Level Emphasis.
  lzgle = np.sum(
    # Combine large zone and low gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  # Large Zone High Gray Level Emphasis.
  lzhge = np.sum(
    # Combine large zone and high gray level emphasis.
    szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN  # Normalize by the total number of zones.

  return {
    "Small Zone Emphasis (SZE)"                  : sze,
    "Large Zone Emphasis (LZE)"                  : lze,
    "Gray Level Non-Uniformity (GLN)"            : gln,
    "Zone Size Non-Uniformity (ZSN)"             : zsn,
    "Zone Percentage (ZP)"                       : zp,
    "Gray Level Variance (GLV)"                  : glv,
    "Zone Size Variance (ZSV)"                   : zsv,
    "Zone Size Entropy (ZSE)"                    : zse,
    "Low Gray Level Zone Emphasis (LGZE)"        : lgze,
    "High Gray Level Zone Emphasis (HGZE)"       : hgze,
    "Small Zone Low Gray Level Emphasis (SZLGE)" : szlge,
    "Small Zone High Gray Level Emphasis (SZHGE)": szhge,
    "Large Zone Low Gray Level Emphasis (LZGLE)" : lzgle,
    "Large Zone High Gray Level Emphasis (LZHGE)": lzhge,
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

# Load and preprocess 3D medical imaging data.
volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

# Set the connectivity type (6 or 26).
C = 26

# Calculate the Size-Zone Matrix.
szMatrix, szDict, N, Z = CalculateGLSZM3DSizeZoneMatrix(
  volumeCropped,
  connectivity=C,
  isNorm=True,
  ignoreZeros=True,
)

# Compute features from Size-Zone Matrix.
features = CalculateGLSZMFeatures(szMatrix, volumeCropped, N, Z)

# Print the connectivity value.
print(f"At connectivity = {C}:")
for feature, value in features.items():
  # Print feature name with value rounded to 4 decimal places.
  print(f"{feature} : {np.round(value, 4)}")
