'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 29th, 2024
# Last Modification Date: Jan 21st, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np

# Define parameters for the GLCM calculation.
d = 1  # Distance between pixel pairs.
theta = 0  # Angle (in degrees) for the direction of pixel pairs.
isSymmetric = False  # Whether to make the GLCM symmetric.

# Define the input matrix (image).
A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]
A = np.array(A)  # Convert the list to a NumPy array.

# Calculate the number of unique intensity levels in the matrix.
N = np.max(A) + 1  # Maximum intensity value + 1.

# Calculate the maximum value in the input matrix.
N = np.max(A) + 1  # Number of unique intensity levels.

# Initialize the co-occurrence matrix with zeros.
coMatrix = np.zeros((N, N))  # Create an N x N matrix filled with zeros.

# Convert the angle from degrees to radians.
theta = np.radians(theta)  # Convert theta to radians for trigonometric calculations.

# Calculate the co-occurrence matrix.
for xLoc in range(A.shape[1]):  # Loop through columns.
  for yLoc in range(A.shape[0]):  # Loop through rows.
    startLoc = (yLoc, xLoc)  # Current pixel location (row, column).

    # Calculate the target pixel location based on distance and angle.
    xTarget = xLoc + np.round(d * np.cos(theta))  # Target column.
    yTarget = yLoc - np.round(d * np.sin(theta))  # Target row.
    endLoc = (int(yTarget), int(xTarget))  # Target pixel location.

    # Check if the target location is within the bounds of the matrix.
    if (
      (endLoc[0] < 0)  # Target row is above the top edge.
      or (endLoc[0] >= A.shape[0])  # Target row is below the bottom edge.
      or (endLoc[1] < 0)  # Target column is to the left of the left edge.
      or (endLoc[1] >= A.shape[1])  # Target column is to the right of the right edge.
    ):
      continue  # Skip this pair if the target is out of bounds.

    # Increment the co-occurrence matrix at the corresponding location.
    coMatrix[A[endLoc], A[startLoc]] += 1  # Increment the count for the pair (start, end).

# If symmetric, add the transpose of the co-occurrence matrix to itself.
if (isSymmetric):
  coMatrix += coMatrix.T  # Make the GLCM symmetric.

# Calculate the energy of the co-occurrence matrix.
energy = np.sum(coMatrix ** 2)  # Sum of the squares of all elements in the GLCM.

# Calculate the contrast in the direction of theta.
contrast = 0.0  # Initialize contrast.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    contrast += (i - j) ** 2 * coMatrix[i, j]  # Weighted sum of squared differences.

# Calculate the homogeneity of the co-occurrence matrix.
homogeneity = 0.0  # Initialize homogeneity.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

# Calculate the entropy of the co-occurrence matrix.
entropy = 0.0  # Initialize entropy.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    if coMatrix[i, j] > 0:  # Check if the value is greater than zero.
      entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

# Calculate the correlation of the co-occurrence matrix.
meanX = 0.0  # Initialize mean of rows.
meanY = 0.0  # Initialize mean of columns.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
    meanY += j * coMatrix[i, j]  # Weighted sum of column indices.

stdDevX = 0.0  # Initialize standard deviation of rows.
stdDevY = 0.0  # Initialize standard deviation of columns.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    stdDevX += (i - meanX) ** 2 * coMatrix[i, j]  # Weighted sum of squared row differences.
    stdDevY += (j - meanY) ** 2 * coMatrix[i, j]  # Weighted sum of squared column differences.

correlation = 0.0  # Initialize correlation.
stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    correlation += (
      (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)
    )  # Weighted sum of normalized differences.

# Calculate the dissimilarity of the co-occurrence matrix.
dissimilarity = 0.0  # Initialize dissimilarity.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    dissimilarity += np.abs(i - j) * coMatrix[i, j]  # Weighted sum of absolute differences.

# Print the results.
print("Energy: ", energy)  # Print the energy of the GLCM.
print("Contrast: ", contrast)  # Print the contrast of the GLCM.
print("Homogeneity: ", homogeneity)  # Print the homogeneity of the GLCM.
print("Entropy: ", entropy)  # Print the entropy of the GLCM.
print("Correlation: ", correlation)  # Print the correlation of the GLCM.
print("Dissimilarity: ", dissimilarity)  # Print the dissimilarity of the GLCM.
