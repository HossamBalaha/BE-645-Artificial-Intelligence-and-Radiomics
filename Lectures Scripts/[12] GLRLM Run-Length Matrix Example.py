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

theta = 45  # Define the angle for scanning direction (0 degrees for horizontal).

# Define a sample 5x5 matrix with gray level values.
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

print("Matrix:")  # Print the original matrix.
print(A)

# Determine the number of unique intensity levels in the matrix.
minA = np.min(A)  # Minimum intensity value.
maxA = np.max(A)  # Maximum intensity value.
N = maxA - minA + 1  # Number of unique intensity levels.
R = np.max(A.shape)  # Determine the maximum dimension of the matrix.
theta = np.radians(theta)  # Convert the angle from degrees to radians.

# Initialize the run-length matrix with zeros.
rlMatrix = np.zeros((N, R))

# Initialize a matrix to track visited pixels.
seenMatrix = np.zeros(A.shape)

# Calculate the direction vector for scanning based on the angle.
# Negative sign is used to ensure the direction is consistent with the angle.
dx = -int(np.round(np.cos(theta)))  # Horizontal direction component.
dy = int(np.round(np.sin(theta)))  # Vertical direction component.

# Iterate over each pixel in the matrix.
for i in range(A.shape[0]):  # Loop through rows.
  for j in range(A.shape[1]):  # Loop through columns.
    # Skip the pixel if it has already been processed.
    if (seenMatrix[i, j] == 1):
      continue

    seenMatrix[i, j] = 1  # Mark the current pixel as seen.
    currentPixel = A[i, j]  # Get the gray level value of the current pixel.
    d = 1  # Initialize the distance (run length) counter.

    # Check consecutive pixels in the specified direction.
    while (
      (i + d * dy >= 0) and  # Ensure the row index is within bounds.
      (i + d * dy < A.shape[0]) and  # Ensure the row index is within bounds.
      (j + d * dx >= 0) and  # Ensure the column index is within bounds.
      (j + d * dx < A.shape[1])  # Ensure the column index is within bounds.
    ):
      # If the next pixel has the same gray level value.
      if (A[i + d * dy, j + d * dx] == currentPixel):
        seenMatrix[int(i + d * dy), int(j + d * dx)] = 1  # Mark it as seen.
        d += 1  # Increment the run length.
      else:
        break  # Stop if the gray level changes.

    # Update the run-length matrix.
    # (- minA) is added to work with matrices that does not start from 0.
    rlMatrix[currentPixel - minA, d - 1] += 1

print("Run-Length Matrix:")  # Print the resulting run-length matrix.
print(rlMatrix)
