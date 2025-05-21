'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 2nd, 2024
# Last Modification Date: Jan 28th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


def CalculateGLRLMRunLengthMatrix(matrix, theta, isNorm=True, ignoreZeros=True):
  """
  Calculate the Gray-Level Run-Length Matrix (GLRLM) for a given 2D matrix.

  The GLRLM is a statistical tool used to quantify the texture of an image by
  analyzing the runs of pixels with the same intensity level in a specific direction.

  Parameters:
  -----------
  matrix : numpy.ndarray
      A 2D matrix representing the image or data for which the GLRLM is to be calculated.

  theta : float
      The angle (in radians) specifying the direction in which runs are to be analyzed.
      The direction is determined by the cosine and sine of this angle.

  isNorm : bool, optional (default=True)
      If True, the resulting GLRLM is normalized by dividing by the total number of runs.
      Normalization ensures that the matrix represents probabilities rather than counts.

  ignoreZeros : bool, optional (default=True)
      If True, runs with zero intensity are ignored in the calculation of the GLRLM.
      This is useful when zero values represent background or irrelevant data.

  Returns:
  --------
  rlMatrix : numpy.ndarray
      A 2D matrix representing the Gray-Level Run-Length Matrix. The rows correspond to
      intensity levels, and the columns correspond to run lengths. If `isNorm` is True,
      the matrix is normalized.
  """

  # Determine the minimum intensity value present in the input matrix.
  minA = np.min(matrix)
  # Determine the maximum intensity value present in the input matrix.
  maxA = np.max(matrix)
  # Calculate the total number of distinct gray levels in the matrix.
  N = maxA - minA + 1
  # Determine maximum possible run length based on image dimensions.
  R = np.max(matrix.shape)

  # Initialize a matrix to count runs of each gray level and length.
  rlMatrix = np.zeros((N, R))
  # Create a binary matrix to track processed pixels to avoid double-counting.
  seenMatrix = np.zeros(matrix.shape)
  # Compute x-direction movement using cosine of theta (negative for coordinate consistency).
  dx = -int(np.round(np.cos(theta)))
  # Compute y-direction movement using sine of theta.
  dy = int(np.round(np.sin(theta)))

  # Process each pixel in the matrix along the y-axis (rows).
  for i in range(matrix.shape[0]):
    # Process each pixel in the matrix along the x-axis (columns).
    for j in range(matrix.shape[1]):
      # Skip already processed pixels to prevent redundant counting.
      if (seenMatrix[i, j] == 1):
        continue

      # Mark current pixel as processed in the tracking matrix.
      seenMatrix[i, j] = 1
      # Store the intensity value of the current pixel.
      currentPixel = matrix[i, j]
      # Initialize run length counter for current pixel's streak.
      d = 1

      # Investigate consecutive pixels in specified direction until boundary or value change.
      while (
        (i + d * dy >= 0) and
        (i + d * dy < matrix.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < matrix.shape[1])
      ):
        # Check if next pixel in direction matches current intensity.
        if (matrix[i + d * dy, j + d * dx] == currentPixel):
          # Mark matching pixel as processed.
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          # Increment run length counter for continued streak.
          d += 1
        else:
          # Break loop when streak ends (different value encountered).
          break

      # Skip recording zero-intensity runs if configured to ignore them.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      # Update GLRLM by incrementing count for current gray level-run length pair.
      # (Adjust gray level index by subtracting minimum value for matrix alignment)
      rlMatrix[currentPixel - minA, d - 1] += 1

  # Normalize matrix to probabilities by dividing by total runs if requested.
  if (isNorm):
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  # Return the computed Gray-Level Run-Length Matrix.
  return rlMatrix


def CalculateGLRLMFeatures(rlMatrix, image):
  """
  Calculate texture features from a Gray-Level Run-Length Matrix (GLRLM).

  This function computes various texture features based on the GLRLM, which is derived
  from an image. These features are commonly used in texture analysis and image processing.

  Parameters:
  -----------
  rlMatrix : numpy.ndarray
      A 2D Gray-Level Run-Length Matrix (GLRLM) computed from an image. The rows represent
      intensity levels, and the columns represent run lengths.

  image : numpy.ndarray
      The original 2D image from which the GLRLM was derived. This is used to determine
      the number of gray levels and the total number of pixels.

  Returns:
  --------
  features : dict
      A dictionary containing the following texture features:
      - "Short Run Emphasis"          : Emphasizes short runs in the image.
      - "Long Run Emphasis"           : Emphasizes long runs in the image.
      - "Gray Level Non-Uniformity"   : Measures the variability of gray levels.
      - "Run Length Non-Uniformity"   : Measures the variability of run lengths.
      - "Run Percentage"              : Ratio of runs to the total number of pixels.
      - "Low Gray Level Run Emphasis" : Emphasizes runs with low gray levels.
      - "High Gray Level Run Emphasis": Emphasizes runs with high gray levels.
  """

  # Determine minimum intensity value in the original image.
  minA = np.min(image)
  # Determine maximum intensity value in the original image.
  maxA = np.max(image)
  # Calculate total number of distinct gray levels in the image.
  N = maxA - minA + 1
  # Get maximum possible run length from image dimensions.
  R = np.max(image.shape)

  # Calculate total number of runs recorded in the GLRLM.
  rlN = np.sum(rlMatrix)

  # Calculate Short Run Emphasis (SRE) emphasizing shorter runs through inverse squared weighting.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Long Run Emphasis (LRE) emphasizing longer runs through squared weighting.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Calculate Gray Level Non-Uniformity (GLN) measuring gray level distribution consistency.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Row sums squared
  ) / rlN

  # Calculate Run Length Non-Uniformity (RLN) measuring run length distribution consistency.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Column sums squared
  ) / rlN

  # Calculate Run Percentage (RP) indicating proportion of image occupied by runs.
  rp = rlN / np.prod(image.shape)

  # Calculate Low Gray Level Run Emphasis (LGRE) weighting low intensities more heavily.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Calculate High Gray Level Run Emphasis (HGRE) weighting high intensities more heavily.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # Package computed features into a dictionary with descriptive keys.
  return {
    "Total Runs"                         : rlN,
    "Short Run Emphasis (SRE)"           : sre,
    "Long Run Emphasis (LRE)"            : lre,
    "Gray Level Non-Uniformity (GLN)"    : gln,
    "Run Length Non-Uniformity (RLN)"    : rln,
    "Run Percentage (RP)"                : rp,
    "Low Gray Level Run Emphasis (LGRE)" : lgre,
    "High Gray Level Run Emphasis (HGRE)": hgre,
  }


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

# Set analysis angle in degrees for run-length direction.
theta = 0
# Convert angle to radians for trigonometric function compatibility.
thetaRad = np.radians(theta)

# Compute GLRLM using cropped ROI and specified parameters.
rlMatrix = CalculateGLRLMRunLengthMatrix(cropped, thetaRad, isNorm=True, ignoreZeros=True)

# Extract texture features from the computed GLRLM matrix.
features = CalculateGLRLMFeatures(rlMatrix, cropped)

# Print header with current analysis angle in degrees.
print(f"At angle {theta} degrees:")
# Iterate through computed features and print formatted values.
for key in features:
  # Print feature name with value rounded to 4 decimal places.
  print(key, ":", np.round(features[key], 4))
