'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np
from HMB_Spring_2026_Helpers import *

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
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 5, 6, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 6, 5],
#   [4, 3, 1, 2, 1]
# ]


# Convert the input list to a NumPy array for easier manipulation.
A = np.array(A)

# Set the connectivity type (4 or 8).
C = 4  # Change to 4 for 4-connectivity or 8 for 8-connectivity.

# Calculate the size-zone matrix for the input matrix using C-connectivity.
szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(A, connectivity=C)

# Compute features from Size-Zone Matrix.
features = CalculateGLSZMFeatures(szMatrix, A, N, Z)

# Print the connectivity value.
print(f"At connectivity = {C}:")
# Iterate through computed features and print formatted values.
for key in features:
  # Print feature name with value rounded to 4 decimal places.
  print(f"{key}:", np.round(features[key], 4))
