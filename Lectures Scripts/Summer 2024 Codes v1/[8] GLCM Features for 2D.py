# Author: Hossam Magdy Balaha
# Date: May 29th, 2024
# Permissions and Citation: Refer to the README file.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def CalculateGLCMCooccuranceMatrix(image, d, theta, isNorm=True, ignoreZeros=True):
  N = np.max(image) + 1

  coMatrix = np.zeros((N, N))

  for xLoc in range(image.shape[1]):
    for yLoc in range(image.shape[0]):
      startLoc = (yLoc, xLoc)
      xTarget = xLoc + np.round(d * np.cos(theta))
      yTarget = yLoc - np.round(d * np.sin(theta))
      endLoc = (int(yTarget), int(xTarget))

      # Check if the target location is within the bounds of the matrix.
      if ((endLoc[0] < 0) or (endLoc[0] >= image.shape[0]) or
        (endLoc[1] < 0) or (endLoc[1] >= image.shape[1])):
        continue

      if (ignoreZeros):
        # Skip the calculation if the pixel values are zero.
        if (image[endLoc] == 0) or (image[startLoc] == 0):
          continue

      # Increment the co-occurrence matrix.
      coMatrix[image[endLoc], image[startLoc]] += 1

  if (isNorm):
    # Normalize the co-occurrence matrix.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix


def CalculateGLCMFeatures(coMatrix):
  N = coMatrix.shape[0]

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)

  # Calculate the contrast in the direction of theta.
  contrast = 0.0
  for i in range(N):
    for j in range(N):
      contrast += (i - j) ** 2 * coMatrix[i, j]

  # Calculate the homogeneity of the co-occurrence matrix.
  homogeneity = 0.0
  for i in range(N):
    for j in range(N):
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)

  # Calculate the entropy of the co-occurrence matrix.
  entropy = 0.0
  for i in range(N):
    for j in range(N):
      if (coMatrix[i, j] > 0):
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])

  # Calculate the correlation of the co-occurrence matrix.
  meanX = 0.0
  meanY = 0.0
  for i in range(N):
    for j in range(N):
      meanX += i * coMatrix[i, j]
      meanY += j * coMatrix[i, j]

  stdDevX = 0.0
  stdDevY = 0.0
  for i in range(N):
    for j in range(N):
      stdDevX += (i - meanX) ** 2 * coMatrix[i, j]
      stdDevY += (j - meanY) ** 2 * coMatrix[i, j]

  correlation = 0.0
  stdDevX = np.sqrt(stdDevX)
  stdDevY = np.sqrt(stdDevY)
  for i in range(N):
    for j in range(N):
      correlation += (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)

  # Calculate the dissimilarity of the co-occurrence matrix.
  dissimilarity = 0.0
  for i in range(N):
    for j in range(N):
      dissimilarity += np.abs(i - j) * coMatrix[i, j]

  return {
    "Energy"       : energy,
    "Contrast"     : contrast,
    "Homogeneity"  : homogeneity,
    "Entropy"      : entropy,
    "Correlation"  : correlation,
    "Dissimilarity": dissimilarity,
    "MeanX"        : meanX,
    "MeanY"        : meanY,
    "StdDevX"      : stdDevX,
    "StdDevY"      : stdDevY,
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

d = 1
theta = 0
theta = np.radians(theta)

coMatrix = CalculateGLCMCooccuranceMatrix(cropped, d, theta, isNorm=True, ignoreZeros=True)
features = CalculateGLCMFeatures(coMatrix)

# Print the GLCM features.
for key in features:
  print(key, ":", features[key])

# Display the co-occurrence matrix.
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Cropped Image")
plt.axis("off")
plt.colorbar()
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.imshow(coMatrix, cmap="gray")
plt.title("Co-occurrence Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()
