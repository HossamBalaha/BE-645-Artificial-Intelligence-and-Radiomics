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

# Define analysis angle in degrees for run-length direction calculation.
theta = 0

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

# Alternative sample matrix (commented out for current demonstration).
# A = [
#   [1, 1, 1, 1, 1],
#   [1, 1, 2, 1, 1],
#   [1, 1, 2, 1, 1],
#   [1, 1, 1, 1, 1],
#   [1, 1, 1, 1, 1],
# ]

# Convert Python list to NumPy array for matrix operations.
A = np.array(A)

# Convert analysis angle from degrees to radians for trigonometric functions.
thetaRad = np.radians(theta)

# Compute GLRLM with normalization disabled and zero values included.
rlMatrix = CalculateGLRLMRunLengthMatrix(A, thetaRad, isNorm=False, ignoreZeros=False)

# Extract texture features from the computed GLRLM matrix.
features = CalculateGLRLMFeatures(rlMatrix, A)

# Print header with current analysis angle in degrees.
print(f"At angle {theta} degrees:")
# Iterate through computed features and print formatted values.
for key in features:
  # Print feature name with value rounded to 4 decimal places.
  print(f"{key}:", np.round(features[key], 4))
