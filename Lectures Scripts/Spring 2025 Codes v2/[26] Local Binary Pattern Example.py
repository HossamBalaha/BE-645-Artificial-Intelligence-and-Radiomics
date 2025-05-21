'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 20th, 2024
# Last Modification Date: Feb 17th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np
import matplotlib.pyplot as plt


def BuildLBPKernel(
  distance=1,  # Distance parameter to determine the size of the kernel.
  theta=135,  # Angle parameter to rotate the kernel (default is 135 degrees).
  isClockwise=False,  # Direction of rotation (False means counterclockwise).
):
  """
  Build a kernel matrix for Local Binary Pattern (LBP) computation.
  The kernel is generated based on the specified distance and angle (theta).
  The kernel is a square matrix of size (2 * distance + 1) x (2 * distance + 1).
  The kernel is filled with powers of 2, representing the weights of the pixels
  in the LBP computation. The kernel is rotated by the specified angle (theta)
  in a clockwise or counterclockwise direction.

  Args:
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the kernel rotation.
    isClockwise (bool): Direction of rotation (True for clockwise, False for counterclockwise).

  Returns:

  """

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")

  # Calculate the total number of elements on the edges.
  noOfElements = 8 * distance  # Total number of edge elements is 8 * distance.

  # Calculate the angle between consecutive elements.
  angle = 360.0 / float(noOfElements)  # Divide 360 degrees by the total number of edge elements.

  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")

  # Check if the angle (theta) is not a multiple of (angle) degrees, raising a ValueError if true.
  if (theta % angle != 0):
    raise ValueError("Theta must be a multiple of the angle between elements.")

  # Calculate the size of the matrix.
  n = 2 * distance + 1  # The size of the kernel is (2 * distance + 1) x (2 * distance + 1).

  # Initialize the matrix with zeros.
  kernel = np.zeros((n, n), dtype=np.uint32)  # Create a zero-filled matrix of size n x n.

  # Generate the coordinates for the edges of the kernel in a clockwise order.
  coordinates = []  # List to store the edge coordinates of the kernel.

  # Add coordinates for the leftmost column (top to bottom).
  for row in range(0, n):  # Iterate over rows from top to bottom.
    coordinates.append((row, 0))  # Append (row, 0) for the leftmost column.

  # Add coordinates for the bottommost row (left to right).
  for col in range(0, n):  # Iterate over columns from left to right.
    coordinates.append((n - 1, col))  # Append (n-1, col) for the bottommost row.

  # Add coordinates for the rightmost column (bottom to top).
  for row in range(n - 1, -1, -1):  # Iterate over rows from bottom to top.
    coordinates.append((row, n - 1))  # Append (row, n-1) for the rightmost column.

  # Add coordinates for the topmost row (right to left).
  for col in range(n - 1, -1, -1):  # Iterate over columns from right to left.
    coordinates.append((0, col))  # Append (0, col) for the topmost row.

  # Remove the repeated coordinates.
  for i in range(len(coordinates) - 1, 0, -1):  # Iterate from the end to the beginning.
    if (coordinates[i] == coordinates[i - 1]):  # Check if the current coordinate is equal to the previous one.
      coordinates.pop(i)  # Remove the current coordinate if it is a duplicate.
  # Remove the last coordinate if it is equal to the first one.
  if (coordinates[-1] == coordinates[0]):  # Check if the last coordinate is equal to the first one.
    coordinates.pop(-1)  # Remove the last coordinate if it is a duplicate.

  # Calculate the shift required to rotate the kernel by the given theta.
  thetaShift = int((theta - 135) / angle)  # Determine how many positions to shift based on theta.

  # Rotate the coordinates list by thetaShift positions.
  coordinates = coordinates[thetaShift:] + coordinates[:thetaShift]  # Shift the coordinates list.

  # Assign powers of 2 to the edge elements in the kernel.
  counter = 0  # Counter to track the current power of 2.

  # Iterate through the shifted coordinates and assign values to the kernel.
  for i in range(len(coordinates)):  # Loop through all edge coordinates.
    x = coordinates[i][0]  # Extract the x-coordinate.
    y = coordinates[i][1]  # Extract the y-coordinate.
    if (kernel[y, x] == 0):  # Check if the position is still zero (not yet assigned).
      kernel[y, x] = 2 ** counter  # Assign 2^counter to the current position.
      counter += 1  # Increment the counter for the next power of 2.

  # If the rotation direction is not clockwise, rotate the kernel counterclockwise.
  if (not isClockwise):
    kernel = kernel.T

  return kernel  # Return the final kernel matrix.


def LocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  """
  Compute the Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The LBP is a texture descriptor that encodes local patterns in the image,
  making it useful for various image analysis tasks.

  Args:
    matrix (np.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    np.ndarray: LBP matrix with the same shape as the input image, containing LBP values.
  """

  # Check if the distance is less than 1, raising a ValueError if true.
  if (distance < 1):
    raise ValueError("Distance must be greater than or equal to 1.")
  # Check if the distance exceeds half of the image dimensions, raising a ValueError if true.
  if (distance > matrix.shape[0] // 2 or distance > matrix.shape[1] // 2):
    raise ValueError("Distance must be less than half of the matrix dimensions.")
  # Check if the angle (theta) is outside the valid range (0 to 360 degrees), raising a ValueError if true.
  if (theta < 0 or theta > 360):
    raise ValueError("Theta must be between 0 and 360 degrees.")
  # Check if the angle (theta) is not a multiple of 45 degrees, raising a ValueError if true.
  if (theta % 45 != 0):
    raise ValueError("Theta must be a multiple of 45 degrees.")

  # Calculate the size of the kernel window based on the distance parameter.
  windowSize = distance * 2 + 1
  # Determine the center coordinates of the kernel window.
  centerX = windowSize // 2
  centerY = windowSize // 2

  # Build the LBP kernel using the specified parameters.
  kernel = BuildLBPKernel(
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
  )

  # Initialize an empty matrix to store the computed LBP values.
  lbpMatrix = np.zeros(matrix.shape, dtype=np.uint32)
  # Pad the input matrix with zeros to handle boundary conditions during convolution.
  paddedA = np.pad(matrix, distance, mode="constant", constant_values=0)

  # Iterate through each pixel in the input matrix to compute its LBP value.
  for y in range(distance, matrix.shape[0] + distance):
    for x in range(distance, matrix.shape[1] + distance):
      # Extract the region of interest (ROI) around the current pixel.
      region = paddedA[
               y - distance:y + distance + 1,
               x - distance:x + distance + 1
               ]
      # Compare each pixel in the ROI with the center pixel to create a binary mask.
      comp = region >= region[centerY, centerX]
      # Compute the LBP value for the current pixel by summing the weighted kernel values.
      lbpMatrix[y - distance, x - distance] = np.sum(kernel[comp])

  # Normalize the LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum LBP values in the matrix.
    minValue = np.min(lbpMatrix)
    maxValue = np.max(lbpMatrix)
    # Normalize the LBP values to the range [0, 255].
    lbpMatrix = ((lbpMatrix - minValue) / (maxValue - minValue) * 255)
    # Ensure the LBP matrix is of type uint8.
    lbpMatrix = lbpMatrix.astype(np.uint8)

  # Return the computed LBP matrix.
  return lbpMatrix


# Define the input matrix A containfing integer values.
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
