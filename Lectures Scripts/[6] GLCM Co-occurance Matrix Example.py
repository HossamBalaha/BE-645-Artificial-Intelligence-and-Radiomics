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
import matplotlib.pyplot as plt
from HMB_Spring_2026_Helpers import CalculateGLCMCooccuranceMatrix

# Define parameters for the co-occurrence matrix calculation.
d = 1  # Distance between pixel pairs.
theta = 0  # Angle (in degrees) for the direction of pixel pairs.
isSymmetric = False  # Whether to make the co-occurrence matrix symmetric.

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
  verbose=True,  # Whether to print verbose output.
)

# Display the co-occurrence matrix as a grayscale image.
plt.figure()  # Create a new figure.
plt.imshow(coMatrix, cmap="gray")  # Display the co-occurrence matrix in grayscale.
plt.title("Co-occurrence Matrix")  # Set the title of the plot.
plt.colorbar()  # Add a color bar to show intensity values.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
