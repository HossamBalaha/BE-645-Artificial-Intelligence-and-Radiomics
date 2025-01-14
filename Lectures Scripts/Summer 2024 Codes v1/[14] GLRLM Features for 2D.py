# Author: Hossam Magdy Balaha
# Date: June 2nd, 2024
# Permissions and Citation: Refer to the README file.

import cv2
import numpy as np


def CalculateGLRLMRunLengthMatrix(image, theta, isNorm=True, ignoreZeros=True):
  N = np.max(image) + 1
  R = np.max(image.shape)

  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(image.shape)
  dx = int(np.round(np.cos(theta)))
  dy = int(np.round(np.sin(theta)))

  for i in range(image.shape[0]):  # Rows
    for j in range(image.shape[1]):  # Columns
      # Skip if already seen.
      if (seenMatrix[i, j] == 1):
        continue

      seenMatrix[i, j] = 1  # Mark as seen.
      currentPixel = image[i, j]  # Current pixel value.
      d = 1  # Distance.

      while (
        (i + d * dy >= 0) and
        (i + d * dy < image.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < image.shape[1])
      ):
        if (image[i + d * dy, j + d * dx] == currentPixel):
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          d += 1
        else:
          break

      # Ignore zeros if needed.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      rlMatrix[currentPixel, d - 1] += 1

  if (isNorm):
    # Normalize the run-length matrix.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix


def CalculateGLRLMFeatures(rlMatrix, image):
  N = np.max(image) + 1
  R = np.max(image.shape)

  rlN = np.sum(rlMatrix)

  # Short Run Emphasis.
  sre = np.sum(
    rlMatrix / (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Long Run Emphasis.
  lre = np.sum(
    rlMatrix * (np.arange(1, R + 1) ** 2),
  ).sum() / rlN

  # Gray Level Non-Uniformity.
  gln = np.sum(
    np.sum(rlMatrix, axis=1) ** 2,  # Sum of each row.
  ) / rlN

  # Run Length Non-Uniformity.
  rln = np.sum(
    np.sum(rlMatrix, axis=0) ** 2,  # Sum of each column.
  ) / rlN

  # Run Percentage.
  rp = rlN / np.prod(image.shape)

  # Low Gray Level Run Emphasis.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # High Gray Level Run Emphasis.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  return {
    "Short Run Emphasis"          : sre,
    "Long Run Emphasis"           : lre,
    "Gray Level Non-Uniformity"   : gln,
    "Run Length Non-Uniformity"   : rln,
    "Run Percentage"              : rp,
    "Low Gray Level Run Emphasis" : lgre,
    "High Gray Level Run Emphasis": hgre,
  }


caseImgPath = r"Sample Liver Image.bmp"
caseSegPath = r"Sample Liver Segmentation.bmp"

# Load the images.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Extract the ROI.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Crop the ROI.
x, y, w, h = cv2.boundingRect(roi)
cropped = roi[y:y + h, x:x + w]

theta = 0
theta = np.radians(theta)

rlMatrix = CalculateGLRLMRunLengthMatrix(cropped, theta, isNorm=True, ignoreZeros=True)
features = CalculateGLRLMFeatures(rlMatrix, cropped)

# Print the GLRLM features.
for key in features:
  print(key, ":", features[key])
