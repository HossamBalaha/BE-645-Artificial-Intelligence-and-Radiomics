'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 12th, 2024
# Last Modification Date: Jun 13th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np
from HMB_Summer_2025_Helpers import *

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
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 4, 3, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 4, 2],
#   [4, 3, 1, 2, 4]
# ]
# A = [
#   [1, 1, 1, 1, 1],
#   [2, 3, 3, 3, 1],
#   [2, 4, 5, 3, 1],
#   [2, 4, 4, 3, 1],
#   [2, 2, 2, 2, 1]
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
