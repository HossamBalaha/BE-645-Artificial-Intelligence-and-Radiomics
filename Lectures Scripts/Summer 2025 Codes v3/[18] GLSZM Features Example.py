'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 13th, 2024
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

# Example input matrix for testing.
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 4, 3, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 4, 2],
#   [4, 3, 1, 2, 4]
# ]

# Convert the input list to a NumPy array for easier manipulation.
A = np.array(A)

# Set the connectivity type (4 or 8).
C = 8

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
