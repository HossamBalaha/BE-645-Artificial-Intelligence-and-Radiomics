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
import matplotlib.pyplot as plt  # For plotting and visualization.
from HMB_Spring_2026_Helpers import *

# Define the input matrix A containing integer values.
A = [
  [17, 32, 32, 47, 11],
  [37, 21, 22, 18, 4],
  [16, 23, 40, 21, 11],
  [13, 55, 41, 28, 12],
  [23, 42, 22, 13, 10],
]

# Alternatively, you can use the following matrix for testing:
# A = [
#   [1, 2, 2, 2, 1],
#   [4, 4, 3, 2, 5],
#   [1, 3, 4, 2, 5],
#   [4, 3, 3, 4, 2],
#   [4, 3, 1, 2, 4],
# ]

# Convert the input matrix A into a NumPy array with unsigned 8-bit integers.
A = np.array(A, dtype=np.uint8)

# Set the distance parameter for the Local Binary Pattern (LBP) computation.
distance = 1

# Specify whether the LBP computation should follow a clockwise direction.
isClockwise = True

# Define the starting angle (theta) for the LBP computation, measured in degrees.
thetaDegree = 135  # Start from the top-left corner.

# Flag to normalize the LBP values (default is False).
normalizeLBP = True

# Create a kernel matrix to represent the LBP pattern weights.
lbpMatrix = LocalBinaryPattern2D(
  matrix=A,  # Input matrix for LBP computation.
  distance=distance,  # Distance from the center pixel to the surrounding pixels.
  theta=thetaDegree,  # Angle in degrees for the kernel rotation.
  isClockwise=isClockwise,  # Direction of rotation (True for clockwise, False for counterclockwise).
  normalizeLBP=normalizeLBP,  # Flag to normalize the LBP values (default is False).
)

# Print the original input matrix for reference.
print("Original Matrix:")
print(A)
print()

# Print the computed LBP matrix with the specified parameters.
print(
  f"LBP Matrix using Theta={thetaDegree}, "
  f"Clockwise={isClockwise}, "
  f"Distance={distance}, "
  f"Normalize={normalizeLBP}:"
)
print(lbpMatrix)

# Plot the original matrix and the LBP matrix side by side.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(A, cmap="gray", interpolation="nearest")
plt.title("Original Matrix")
plt.axis("off")  # Hide the axes for better visualization.
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 2, 2)
plt.imshow(lbpMatrix, cmap="gray", interpolation="nearest")
plt.title(f"LBP Matrix (Theta={thetaDegree}, Clockwise={isClockwise})")
plt.axis("off")  # Hide the axes for better visualization.
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.savefig("Data/LBP_Matrix.png", dpi=720, bbox_inches="tight")  # Save the plot as an image.
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
