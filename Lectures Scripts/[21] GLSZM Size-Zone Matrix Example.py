'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 12th, 2024
# Last Modification Date: Feb 11th, 2025
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

# Other examples to try (commented out for now).
# A = [
#   [4, 1, 4, 2, 4],
#   [4, 4, 4, 4, 2],
#   [4, 4, 4, 2, 4],
#   [4, 3, 4, 3, 2],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 3, 4, 4],
#   [3, 2, 2, 2],
#   [4, 1, 4, 1],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 1, 2, 4],
#   [3, 3, 3, 2],
#   [4, 1, 4, 1],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
# ]
# A = [
#   [4, 1, 2, 4],
#   [4, 4, 4, 4],
#   [4, 4, 4, 4],
#   [4, 3, 3, 4],
# ]

# Convert the input list to a NumPy array for easier manipulation.
A = np.array(A)

# Set the connectivity type (4 or 8).
C = 8

# Calculate the size-zone matrix for the input matrix using C-connectivity.
szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(A, connectivity=C)

# Print the original matrix, size-zone dictionary, and size-zone matrix.
print("Matrix:")
print(A)
print("Size-Zone Dictionary:")
print(szDict)
print("Size-Zone Matrix:")
print(szMatrix)
