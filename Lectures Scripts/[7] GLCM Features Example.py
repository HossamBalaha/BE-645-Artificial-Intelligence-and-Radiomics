'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 29th, 2024
# Last Modification Date: Jun 5th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np

# Define parameters for the GLCM calculation.
d = 1  # Distance between pixel pairs.
theta = 0  # Angle (in degrees) for the direction of pixel pairs.
isSymmetric = True  # Whether to make the GLCM symmetric.

# Define the input matrix (image).
A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]
# Second example that starts from 1 not 0.
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 5, 6, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 6, 5],
#   [4, 3, 1, 2, 1],
# ]
A = np.array(A)  # Convert the list to a NumPy array.

# Print the input matrix.
print("Matrix:")
print(A)

# Determine the number of unique intensity levels in the matrix.
minA = np.min(A)  # Minimum intensity value.
maxA = np.max(A)  # Maximum intensity value.
N = maxA - minA + 1  # Number of unique intensity levels.

# Initialize the co-occurrence matrix with zeros.
# Create an N x N matrix filled with zeros.
coMatrix = np.zeros((N, N))

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
    # (- minA) is added to work with matrices that does not start from 0.
    # Increment the count for the pair (start, end).
    # A[startLoc] and A[endLoc] are the intensity values at the start and end locations.
    startPixel = A[startLoc] - minA  # Adjust start pixel value.
    endPixel = A[endLoc] - minA  # Adjust end pixel value.
    coMatrix[endPixel, startPixel] += 1

# If symmetric, add the transpose of the co-occurrence matrix to itself.
if (isSymmetric):
  coMatrix += coMatrix.T  # Make the GLCM symmetric.

# Print the final co-occurrence matrix.
print("Co-occurrence Matrix:")
print(coMatrix)

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
    if (coMatrix[i, j] > 0):  # Check if the value is greater than zero.
      entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

# Calculate the correlation of the co-occurrence matrix.
totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
meanX = 0.0  # Initialize mean of rows.
meanY = 0.0  # Initialize mean of columns.
for i in range(N):  # Loop through rows.
  for j in range(N):  # Loop through columns.
    meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
    meanY += j * coMatrix[i, j]  # Weighted sum of column indices.
meanX /= totalSum  # Calculate mean of rows.
meanY /= totalSum  # Calculate mean of columns.

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
print("Energy:", np.round(energy, 4))  # Print the energy of the GLCM.
print("Contrast:", np.round(contrast, 4))  # Print the contrast of the GLCM.
print("Homogeneity:", np.round(homogeneity, 4))  # Print the homogeneity of the GLCM.
print("Entropy:", np.round(entropy, 4))  # Print the entropy of the GLCM.
print("Dissimilarity:", np.round(dissimilarity, 4))  # Print the dissimilarity of the GLCM.
print("Total Sum:", np.round(totalSum, 4))  # Print the total sum of the GLCM.
print("Mean X:", np.round(meanX, 4))  # Print the mean of rows.
print("Mean Y:", np.round(meanY, 4))  # Print the mean of columns.
print("Standard Deviation X:", np.round(stdDevX, 4))  # Print the standard deviation of rows.
print("Standard Deviation Y:", np.round(stdDevY, 4))  # Print the standard deviation of columns.
print("Correlation:", np.round(correlation, 4))  # Print the correlation of the GLCM.
