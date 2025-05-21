'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 29th, 2024
# Last Modification Date: Jan 23rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np
import matplotlib.pyplot as plt

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

# Determine the number of unique intensity levels in the matrix.
minA = np.min(A)  # Minimum intensity value.
maxA = np.max(A)  # Maximum intensity value.
N = maxA - minA + 1  # Number of unique intensity levels.

# Initialize the co-occurrence matrix with zeros.
coMatrix = np.zeros((N, N))  # Create an N x N matrix filled with zeros.

# Convert the angle from degrees to radians.
theta = np.radians(theta)  # Convert theta to radians for trigonometric calculations.

# Iterate over each pixel in the matrix to calculate the co-occurrence matrix.
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
    coMatrix[A[endLoc] - minA, A[startLoc] - minA] += 1  # Increment the count for the pair (start, end).

    # Print the current pair and the updated co-occurrence matrix.
    print(
      f"Start: {startLoc}, End: {endLoc}",  # Print the start and end locations.
      f"Increment (x={A[startLoc]}, y={A[endLoc]}) by 1."  # Print the intensity values.
    )

# If symmetric, add the transpose of the co-occurrence matrix to itself.
if (isSymmetric):
  coMatrix += coMatrix.T  # Make the co-occurrence matrix symmetric.

# Print the final co-occurrence matrix.
print("Co-occurrence Matrix:")
print(coMatrix)

# Display the co-occurrence matrix as a grayscale image.
plt.figure()  # Create a new figure.
plt.imshow(coMatrix, cmap="gray")  # Display the co-occurrence matrix in grayscale.
plt.title("Co-occurrence Matrix")  # Set the title of the plot.
plt.colorbar()  # Add a color bar to show intensity values.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
