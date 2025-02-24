'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 20th, 2024
# Last Modification Date: Feb 24th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os
import cv2
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


def UniformLocalBinaryPattern2D(
  matrix,
  distance=1,
  theta=135,
  isClockwise=False,
  normalizeLBP=False,
):
  """
  Compute the Uniform Local Binary Pattern (LBP) matrix for a given 2D matrix.
  This function calculates the LBP values based on the specified distance,
  angle (theta), and direction (clockwise or counterclockwise).
  The Uniform LBP is a variant of LBP that focuses on uniform patterns,
  making it useful for texture analysis and classification tasks.
  The uniform patterns are defined as those with at most two transitions
  between 0 and 1 in the binary representation of the LBP value.

  Args:
    matrix (np.ndarray): Input 2D matrix (grayscale) for LBP computation.
    distance (int): Distance from the center pixel to the surrounding pixels.
    theta (int): Angle in degrees for the LBP computation (must be a multiple of 45).
    isClockwise (bool): Direction of LBP computation (True for clockwise, False for counterclockwise).
    normalizeLBP (bool): Flag to normalize the LBP values (default is False).

  Returns:
    np.ndarray: Uniform LBP matrix with the same shape as the input image, containing LBP values.
  """
  # Run the standard LBP function to get the LBP matrix.
  lbpMatrix = LocalBinaryPattern2D(
    matrix,
    distance=distance,
    theta=theta,
    isClockwise=isClockwise,
    normalizeLBP=False,  # No need to normalize here.
  )

  # Initialize an empty matrix to store the uniform LBP values.
  uniformMatrix = np.zeros(matrix.shape, dtype=np.uint32)

  # Iterate through each pixel in the LBP matrix to compute uniform LBP values.
  for y in range(matrix.shape[0]):
    for x in range(matrix.shape[1]):
      # Convert the LBP value to binary representation with 8 * distance bits.
      binary = np.binary_repr(
        lbpMatrix[y, x],
        width=8 * distance,
      )
      # Count the number of transitions (0 to 1 or 1 to 0) in the binary representation.
      transitions = 0
      for i in range(1, len(binary)):
        # Count transitions between consecutive bits.
        if (binary[i] != binary[i - 1]):
          transitions += 1

      # If the number of transitions is less than or equal to 2, assign the LBP value.
      if (transitions <= 2):
        # Assign the LBP value to the uniform matrix.
        uniformMatrix[y, x] = int(binary, 2)

  # Normalize the uniform LBP values if the flag is set to True.
  if (normalizeLBP):
    # Find the minimum and maximum uniform LBP values in the matrix.
    minValue = np.min(uniformMatrix)
    maxValue = np.max(uniformMatrix)
    # Normalize the uniform LBP values to the range [0, 255].
    uniformMatrix = ((uniformMatrix - minValue) / (maxValue - minValue) * 255).astype(np.uint8)

  # Ensure the uniform LBP matrix is of type uint8.
  uniformMatrix = uniformMatrix.astype(np.uint8)

  # Return the computed uniform LBP matrix.
  return uniformMatrix


# Define path to sample liver image file.
caseImgPath = r"Data/Sample Liver Image.bmp"
# Define path to corresponding liver segmentation mask file.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"

# Verify both image and mask files exist before attempting to load.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  # Raise error with descriptive message if any file is missing.
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load input image as grayscale (single channel intensity values).
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
# Load segmentation mask as grayscale (binary or labeled format).
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Extract Region of Interest (ROI) by masking input image with segmentation.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Calculate bounding box coordinates of non-zero region in ROI.
x, y, w, h = cv2.boundingRect(roi)
# Crop ROI to tight bounding box around segmented area.
cropped = roi[y:y + h, x:x + w]

# Validate cropped image contains non-zero pixels to prevent empty processing.
if (np.sum(cropped) <= 0):
  # Raise error if cropped image is completely black/empty.
  raise ValueError("The cropped image is empty. Please check the segmentation mask.")

# Set the parameters for the LBP computation.
distanceLBP = 1  # Distance from the center pixel to the surrounding pixels.
isClockwiseLBP = False  # Direction of LBP computation (True for clockwise, False for counterclockwise).
thetaLBP = 135  # Start from the top-left corner.
normalizeLBP = True  # Flag to normalize LBP values.

# Compute the LBP matrix and uniform LBP matrix for the cropped image.
lbpMatrix = LocalBinaryPattern2D(
  cropped,
  distance=distanceLBP,
  theta=thetaLBP,
  isClockwise=isClockwiseLBP,
  normalizeLBP=normalizeLBP,
)
lbpMatrixUniform = UniformLocalBinaryPattern2D(
  cropped,
  distance=distanceLBP,
  theta=thetaLBP,
  isClockwise=isClockwiseLBP,
  normalizeLBP=normalizeLBP,
)

# Display the original image, LBP image, and uniform LBP image side by side.
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 3, 2)
plt.imshow(lbpMatrix, cmap="gray")
plt.title("LBP Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.subplot(1, 3, 3)
plt.imshow(lbpMatrixUniform, cmap="gray")
plt.title("Uniform LBP Image")
plt.axis("off")
plt.tight_layout()  # Adjust layout to prevent overlap.
plt.savefig("Data/LBP_Images.png", dpi=720, bbox_inches="tight")  # Save the plot as an image.
plt.show()  # Display the plot.
plt.close()  # Close the plot to free up memory.
