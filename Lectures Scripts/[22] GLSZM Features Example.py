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
import numpy as np


def FindConnectedRegions(image, connectivity=4):
  """
  Finds connected regions in a 2D image based on pixel connectivity.

  Parameters:
      image (numpy.ndarray): A 2D NumPy array representing the input image.
                             Each element represents a pixel value.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 4: 4-connectivity (up, down, left, right).
                          - 8: 8-connectivity (includes diagonals).

  Returns:
      dict: A dictionary where keys are unique pixel values from the image,
            and values are lists of sets. Each set contains the coordinates
            (i, j) of pixels belonging to a connected region for that pixel value.
  """

  def RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity=4):
    """
    Recursive helper function to find all connected pixels for a given starting pixel.

    Parameters:
        i (int): Row index of the current pixel.
        j (int): Column index of the current pixel.
        currentPixel (int): The pixel value being processed.
        region (set): A set to store the coordinates of connected pixels.
        seenMatrix (numpy.ndarray): A 2D matrix to track visited pixels.
        connectivity (int): The type of connectivity (4 or 8).

    Returns:
        None: The function modifies the `region` and `seenMatrix` in place.
    """
    # Check if the current pixel is out of bounds, already seen, or not matching the current pixel value.
    if (
      (i < 0) or  # Check if row index is out of bounds.
      (i >= image.shape[0]) or
      (j < 0) or
      (j >= image.shape[1]) or
      (image[i, j] != currentPixel) or  # Check if pixel value matches the current pixel value.
      ((i, j) in region)  # Check if the pixel has already been added to the region.
    ):
      return  # Exit if any condition is met.

    # Add the current pixel to the region and mark it as seen.
    region.add((i, j))
    seenMatrix[i, j] = 1

    # Recursively check the neighboring pixels (up, left, down, right).
    RecursiveHelper(i - 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j - 1, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i + 1, j, currentPixel, region, seenMatrix, connectivity)
    RecursiveHelper(i, j + 1, currentPixel, region, seenMatrix, connectivity)

    # If 8-connectivity is specified, also check diagonal neighbors.
    if (connectivity == 8):
      RecursiveHelper(i - 1, j - 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i - 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      RecursiveHelper(i + 1, j - 1, currentPixel, region, seenMatrix, connectivity)

  # Initialize a matrix to keep track of seen pixels.
  seenMatrix = np.zeros(image.shape)

  # Dictionary to store regions grouped by pixel values.
  regions = {}

  # Iterate over each pixel in the image.
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      # Skip if the pixel has already been processed.
      if (seenMatrix[i, j]):
        continue

      # Get the current pixel value.
      currentPixel = image[i, j]

      # Initialize a list for this pixel value if it doesn't exist.
      if (currentPixel not in regions):
        regions[currentPixel] = []

      # Initialize a new region set for the current pixel.
      region = set()

      # Use the helper function to find all connected pixels.
      RecursiveHelper(i, j, currentPixel, region, seenMatrix, connectivity)

      # Add the region to the dictionary if it contains any pixels.
      if (len(region) > 0):
        regions[currentPixel].append(region)

  # Return the dictionary of regions.
  return regions


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4, isNorm=False, ignoreZeros=False):
  """
  Calculate the Size-Zone Matrix for a given image based on connected regions.

  Parameters:
      image (numpy.ndarray): A 2D NumPy array representing the input image.
                             Each element represents a pixel value.
      connectivity (int): The type of connectivity to use for determining
                          connected regions. Options are:
                          - 4: 4-connectivity (up, down, left, right).
                          - 8: 8-connectivity (includes diagonals).
      isNorm (bool): Whether to normalize the size-zone matrix.
      ignoreZeros (bool): Whether to ignore zero pixel values.

  Returns:
      szMatrix (numpy.ndarray): A 2D NumPy array representing the Size-Zone Matrix.
      szDict (dict): A dictionary where keys are unique pixel values from the image,
                      and values are lists of sets. Each set contains the coordinates
                      (i, j) of pixels belonging to a connected region for that pixel value.
      N (int): The number of unique pixel values in the image.
      Z (int): The maximum size of any region in the dictionary.
  """

  if (image.ndim != 2):
    raise ValueError("The input image must be a 2D array.")

  if (connectivity not in [4, 8]):
    raise ValueError("Connectivity must be either 4 or 8.")

  if (image.size == 0):
    raise ValueError("The input image is empty.")

  if (np.max(image) == 0):
    raise ValueError("The input image is completely black.")

  # Find connected regions in the image.
  szDict = FindConnectedRegions(image, connectivity=connectivity)

  # Determine the number of unique pixel values in the image.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  # Find the maximum size of any region in the dictionary.
  # By iterating over all zones of all pixel values and getting the length of the largest zone.
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])

  # Initialize a size-zone matrix with zeros.
  szMatrix = np.zeros((N, Z))

  # Populate the size-zone matrix with counts of regions for each pixel value.
  for currentPixel, zones in szDict.items():
    for zone in zones:
      # Ignore zeros if needed.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Increment the count for the corresponding pixel value and region size.
      szMatrix[currentPixel - minA, len(zone) - 1] += 1

  szMatrixSum = np.sum(szMatrix)

  if (szMatrixSum == 0):
    return szMatrix, szDict, N, Z

  # Normalize the size-zone matrix if required.
  if (isNorm):
    # Normalize the size-zone matrix.
    szMatrix = szMatrix / np.sum(szMatrix)

  return szMatrix, szDict, N, Z


# Example input matrix.
A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]  # Define a 2D list representing the input image matrix.

# Example input matrix for testing.
A = [
  [1, 2, 2, 2, 1],
  [4, 4, 3, 2, 5],
  [1, 3, 4, 2, 5],
  [4, 3, 3, 4, 2],
  [4, 3, 1, 2, 4]
]

# Convert the input list to a NumPy array for easier manipulation.
A = np.array(A)

# Set the connectivity type (4 or 8).
C = 4

# Calculate the size-zone matrix for the input matrix using C-connectivity.
szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(A, connectivity=C)

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
zp = szN / np.prod(A.shape)

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

# Print the calculated metrics with 4 decimal places.
print("Small Zone Emphasis (SZE):", np.round(sze, 4))  # Print Small Zone Emphasis.
print("Large Zone Emphasis (LZE):", np.round(lze, 4))  # Print Large Zone Emphasis.
print("Gray Level Non-Uniformity (GLN):", np.round(gln, 4))  # Print Gray Level Non-Uniformity.
print("Zone Size Non-Uniformity (ZSN):", np.round(zsn, 4))  # Print Zone Size Non-Uniformity.
print("Zone Percentage (ZP):", np.round(zp, 4))  # Print Zone Percentage.
print("Gray Level Variance (GLV):", np.round(glv, 4))  # Print Gray Level Variance.
print("Zone Size Variance (ZSV):", np.round(zsv, 4))  # Print Zone Size Variance.
print("Zone Size Entropy (ZSE):", np.round(zse, 4))  # Print Zone Size Entropy.
print("Low Gray Level Zone Emphasis (LGZE):", np.round(lgze, 4))  # Print Low Gray Level Zone Emphasis.
print("High Gray Level Zone Emphasis (HGZE):", np.round(hgze, 4))  # Print High Gray Level Zone Emphasis.
print("Small Zone Low Gray Level Emphasis (SZLGE):", np.round(szlge, 4))  # Print Small Zone Low Gray Level Emphasis.
print("Small Zone High Gray Level Emphasis (SZHGE):", np.round(szhge, 4))  # Print Small Zone High Gray Level Emphasis.
print("Large Zone Low Gray Level Emphasis (LZGLE):", np.round(lzgle, 4))  # Print Large Zone Low Gray Level Emphasis.
print("Large Zone High Gray Level Emphasis (LZHGE):", np.round(lzhge, 4))  # Print Large Zone High Gray Level Emphasis.
