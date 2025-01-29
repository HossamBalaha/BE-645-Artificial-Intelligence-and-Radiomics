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
  # Calculate minimum intensity value in the input matrix for intensity range adjustment.
  minA = np.min(matrix)
  # Calculate maximum intensity value in the input matrix for intensity range adjustment.
  maxA = np.max(matrix)
  # Determine total number of distinct gray levels by calculating intensity range span.
  N = maxA - minA + 1
  # Find maximum potential run length based on largest matrix dimension.
  R = np.max(matrix.shape)

  # Initialize empty GLRLM matrix with dimensions (intensity levels × max run length).
  rlMatrix = np.zeros((N, R))
  # Create tracking matrix to prevent duplicate processing of pixels in runs.
  seenMatrix = np.zeros(matrix.shape)
  # Calculate x-direction step using cosine (negative for coordinate system alignment).
  dx = int(np.round(np.cos(theta)))
  # Calculate y-direction step using sine of the analysis angle.
  dy = int(np.round(np.sin(theta)))

  # Adjust direction for specific angles to ensure consistent run direction.
  if (theta in [np.radians(45), np.radians(135)]):
    dx = -dx  # Adjust x-direction for 45 and 135 degrees.
    dy = dy  # Keep y-direction unchanged for 45 and 135 degrees.

  # Iterate through each row index of the input matrix.
  for i in range(matrix.shape[0]):
    # Iterate through each column index of the input matrix.
    for j in range(matrix.shape[1]):
      # Skip already processed pixels to prevent duplicate counting.
      if (seenMatrix[i, j] == 1):
        continue

      # Mark current pixel as processed in tracking matrix.
      seenMatrix[i, j] = 1
      # Store intensity value of current pixel for run comparison.
      currentPixel = matrix[i, j]
      # Initialize run length counter for current streak.
      d = 1

      # Explore consecutive pixels in specified direction until boundary or value change.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        # Check if subsequent pixel matches current intensity value.
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          # Mark matching pixel as processed in tracking matrix.
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          # Increment run length counter for continued streak.
          d += 1
        else:
          # Exit loop when streak breaks (different value encountered).
          break

      # Skip zero-value runs if configured to ignore background.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update GLRLM by incrementing count at corresponding intensity-runlength position.
      # (Adjust intensity index by minimum value for proper matrix positioning)
      rlMatrix[currentPixel - minA, d - 1] += 1

  # Normalize matrix to probability distribution if requested.
  if (isNorm):
    # Add small epsilon to prevent division by zero in empty matrices.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  # Return computed Gray-Level Run-Length Matrix.
  return rlMatrix


# Define analysis angle in degrees for run-length direction calculation.
theta = 90

# Create sample 5x5 matrix with various intensity values for demonstration.
A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]

# Alternative sample matrix (commented out for current demonstration).
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 5, 6, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 6, 5],
#   [4, 3, 1, 2, 1],
# ]

# Convert Python list to NumPy array for matrix operations.
A = np.array(A)
# Calculate minimum intensity value for intensity range adjustment.
minA = np.min(A)
# Calculate maximum intensity value for intensity range adjustment.
maxA = np.max(A)
# Determine total number of distinct intensity levels.
N = maxA - minA + 1
# Get maximum possible run length from matrix dimensions.
R = np.max(A.shape)
# Convert analysis angle from degrees to radians for trigonometric functions.
thetaRad = np.radians(theta)

# Compute GLRLM with normalization disabled and zero values included.
rlMatrix = CalculateGLRLMRunLengthMatrix(A, thetaRad, isNorm=False, ignoreZeros=False)
# Calculate total number of runs in the computed GLRLM.
rlN = np.sum(rlMatrix)

# Calculate Short Run Emphasis (SRE) using inverse squared run length weights.
sre = np.sum(
  rlMatrix / (np.arange(1, R + 1) ** 2),
).sum() / rlN

# Calculate Long Run Emphasis (LRE) using squared run length weights.
lre = np.sum(
  rlMatrix * (np.arange(1, R + 1) ** 2),
).sum() / rlN

# Calculate Gray Level Non-Uniformity (GLN) measuring intensity distribution consistency.
gln = np.sum(
  np.sum(rlMatrix, axis=1) ** 2,
) / rlN

# Calculate Run Length Non-Uniformity (RLN) measuring run length distribution consistency.
rln = np.sum(
  np.sum(rlMatrix, axis=0) ** 2,
) / rlN

# Calculate Run Percentage (RP) indicating runs-to-pixels ratio.
rp = rlN / np.prod(A.shape)

# Calculate Low Gray Level Run Emphasis (LGRE) using inverse squared intensity weights.
lgre = np.sum(
  rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
).sum() / rlN

# Calculate High Gray Level Run Emphasis (HGRE) using squared intensity weights.
hgre = np.sum(
  rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
).sum() / rlN

# Print raw GLRLM matrix for visual inspection.
print("Run-Length Matrix:")
print(rlMatrix)

# Print computed texture features with header showing analysis angle.
print(f"\nAt angle {theta} degrees:")
# Print the total number of runs.
print("Total Runs:", rlN)
# Print SRE value rounded to 4 decimal places.
print("Short Run Emphasis (SRE):", np.round(sre, 4))
# Print LRE value rounded to 4 decimal places.
print("Long Run Emphasis (LRE):", np.round(lre, 4))
# Print GLN value rounded to 4 decimal places.
print("Gray Level Non-Uniformity (GLN):", np.round(gln, 4))
# Print RLN value rounded to 4 decimal places.
print("Run Length Non-Uniformity (RLN):", np.round(rln, 4))
# Print RP value rounded to 4 decimal places.
print("Run Percentage (RP):", np.round(rp, 4))
# Print LGRE value rounded to 4 decimal places.
print("Low Gray Level Run Emphasis (LGRE):", np.round(lgre, 4))
# Print HGRE value rounded to 4 decimal places.
print("High Gray Level Run Emphasis (HGRE):", np.round(hgre, 4))
