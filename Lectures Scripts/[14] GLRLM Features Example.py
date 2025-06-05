'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 2nd, 2024
# Last Modification Date: Jun 5th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np  # For numerical operations.
from HMB_Summer_2025_Helpers import *

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
