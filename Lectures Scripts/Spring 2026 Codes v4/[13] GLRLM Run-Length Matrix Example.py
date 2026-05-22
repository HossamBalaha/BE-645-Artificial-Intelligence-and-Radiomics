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
import numpy as np  # For numerical operations.
from HMB_Spring_2026_Helpers import *

# Define the angle for scanning direction (0 degrees for horizontal).
theta = 0
# Convert angle to radians for trigonometric functions.
thetaRad = np.radians(theta)

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

# Compute GLRLM with normalization disabled and zero values included.
rlMatrix = CalculateGLRLMRunLengthMatrix(A, thetaRad, isNorm=False, ignoreZeros=False)

print("Run-Length Matrix:")  # Print the resulting run-length matrix.
print(rlMatrix)
