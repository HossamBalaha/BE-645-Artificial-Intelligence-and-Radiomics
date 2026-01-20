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
from HMB_Spring_2026_Helpers import CalculateGLCMCooccuranceMatrix, CalculateGLCMFeatures

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

coMatrix = CalculateGLCMCooccuranceMatrix(
  A,  # Input matrix.
  d,  # Distance between pixel pairs.
  theta,  # Angle (in degrees) for the direction of pixel pairs.
  isSymmetric=isSymmetric,  # Whether to make the co-occurrence matrix symmetric.
  isNorm=False,  # Whether to normalize the co-occurrence matrix.
  ignoreZeros=False,  # Whether to ignore zero values in the input matrix.
  verbose=False,  # Whether to print verbose output.
)

features = CalculateGLCMFeatures(coMatrix)  # Calculate GLCM features.

# Extract features from the dictionary.
energy = features["Energy"]
contrast = features["Contrast"]
homogeneity = features["Homogeneity"]
entropy = features["Entropy"]
dissimilarity = features["Dissimilarity"]
totalSum = features["TotalSum"]
meanX = features["MeanX"]
meanY = features["MeanY"]
stdDevX = features["StdDevX"]
stdDevY = features["StdDevY"]
correlation = features["Correlation"]

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
