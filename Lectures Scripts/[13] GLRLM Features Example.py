'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 2nd, 2024
# Last Modification Date: Jan 28th, 2025
# Permissions and Citation: Refer to the README file.
'''

import numpy as np  # Import the NumPy library for numerical operations.


def CalculateGLRLMRunLengthMatrix(matrix, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate the Gray-Level Run-Length Matrix (GLRLM) for a given 2D matrix.

  The GLRLM is a statistical tool used to quantify the texture of an image by
  analyzing the runs of pixels with the same intensity level in a specific direction.

  Parameters:
  -----------
  matrix : numpy.ndarray
      A 2D matrix representing the image or data for which the GLRLM is to be calculated.

  theta : float
      The angle (in radians) specifying the direction in which runs are to be analyzed.
      The direction is determined by the cosine and sine of this angle.

  isNorm : bool, optional (default=True)
      If True, the resulting GLRLM is normalized by dividing by the total number of runs.
      Normalization ensures that the matrix represents probabilities rather than counts.

  ignoreZeros : bool, optional (default=True)
      If True, runs with zero intensity are ignored in the calculation of the GLRLM.
      This is useful when zero values represent background or irrelevant data.

  Returns:
  --------
  rlMatrix : numpy.ndarray
      A 2D matrix representing the Gray-Level Run-Length Matrix. The rows correspond to
      intensity levels, and the columns correspond to run lengths. If `isNorm` is True,
      the matrix is normalized.
  """
  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(matrix)  # Find the minimum intensity value in the matrix.
  maxA = np.max(matrix)  # Find the maximum intensity value in the matrix.
  N = maxA - minA + 1  # Calculate the number of unique intensity levels.
  R = np.max(matrix.shape)  # Determine the maximum possible run length.

  rlMatrix = np.zeros((N, R))  # Initialize the run-length matrix with zeros.
  seenMatrix = np.zeros(matrix.shape)  # Initialize a matrix to track seen pixels.
  # Negative sign is used to ensure the direction is consistent with the angle.
  dx = -int(np.round(np.cos(theta)))  # Calculate the x-direction step based on theta.
  dy = int(np.round(np.sin(theta)))  # Calculate the y-direction step based on theta.

  for i in range(matrix.shape[0]):  # Iterate over each row in the matrix.
    for j in range(matrix.shape[1]):  # Iterate over each column in the matrix.
      # Skip the pixel if it has already been processed.
      if (seenMatrix[i, j] == 1):
        continue

      seenMatrix[i, j] = 1  # Mark the current pixel as seen.
      currentPixel = matrix[i, j]  # Get the intensity value of the current pixel.
      d = 1  # Initialize the run length distance.

      # Check consecutive pixels in the direction specified by theta.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1  # Mark the pixel as seen.
          d += 1  # Increment the run length distance.
        else:
          break  # Stop if the run ends.

      # Skip zero-intensity runs if ignoreZeros is True.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update the run-length matrix.
      # (- minA) is added to work with matrices that does not start from 0.
      rlMatrix[currentPixel - minA, d - 1] += 1

  if (isNorm):
    # Normalize the run-length matrix by dividing by the total number of runs.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix  # Return the computed run-length matrix.


theta = 45  # Set the angle for run-length calculation (0 degrees).

A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]

# Second example.
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 5, 6, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 6, 5],
#   [4, 3, 1, 2, 1],
# ]

# Convert the list to a NumPy array.
A = np.array(A)

# Determine the number of unique intensity levels in the matrix.
minA = np.min(A)  # Find the minimum intensity value in the matrix.
maxA = np.max(A)  # Find the maximum intensity value in the matrix.
N = maxA - minA + 1  # Calculate the number of unique intensity levels.
R = np.max(A.shape)  # Determine the maximum possible run length.
thetaRad = np.radians(theta)  # Convert the angle from degrees to radians.

rlMatrix = CalculateGLRLMRunLengthMatrix(A, thetaRad, isNorm=False, ignoreZeros=False)  # Compute the run-length matrix.

rlN = np.sum(rlMatrix)  # Calculate the total number of runs in the matrix.

# Short Run Emphasis (SRE): Emphasizes shorter runs.
sre = np.sum(
  rlMatrix / (np.arange(1, R + 1) ** 2),  # Weight each run by the inverse square of its length.
).sum() / rlN  # Normalize by the total number of runs.

# Long Run Emphasis (LRE): Emphasizes longer runs.
lre = np.sum(
  rlMatrix * (np.arange(1, R + 1) ** 2),  # Weight each run by the square of its length.
).sum() / rlN  # Normalize by the total number of runs.

# Gray Level Non-Uniformity (GLN): Measures the variability of gray levels.
gln = np.sum(
  np.sum(rlMatrix, axis=1) ** 2,  # Sum of each row (gray level) squared.
) / rlN  # Normalize by the total number of runs.

# Run Length Non-Uniformity (RLN): Measures the variability of run lengths.
rln = np.sum(
  np.sum(rlMatrix, axis=0) ** 2,  # Sum of each column (run length) squared.
) / rlN  # Normalize by the total number of runs.

# Run Percentage (RP): Measures the proportion of runs relative to the total number of pixels.
rp = rlN / np.prod(A.shape)  # Divide the total number of runs by the total number of pixels.

# Low Gray Level Run Emphasis (LGRE): Emphasizes runs with lower gray levels.
lgre = np.sum(
  rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),  # Weight each run by the inverse square of its gray level.
).sum() / rlN  # Normalize by the total number of runs.

# High Gray Level Run Emphasis (HGRE): Emphasizes runs with higher gray levels.
hgre = np.sum(
  rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),  # Weight each run by the square of its gray level.
).sum() / rlN  # Normalize by the total number of runs.

# Print the run-length matrix.
print("Run-Length Matrix:")
print(rlMatrix)

# Print the computed texture features rounded to 4 decimal places.
print(f"At angle {theta} degrees:")
print("Short Run Emphasis:", np.round(sre, 4))
print("Long Run Emphasis:", np.round(lre, 4))
print("Gray Level Non-Uniformity:", np.round(gln, 4))
print("Run Length Non-Uniformity:", np.round(rln, 4))
print("Run Percentage:", np.round(rp, 4))
print("Low Gray Level Run Emphasis:", np.round(lgre, 4))
print("High Gray Level Run Emphasis:", np.round(hgre, 4))
