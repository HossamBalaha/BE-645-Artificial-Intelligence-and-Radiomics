# Author: Hossam Magdy Balaha
# Date: May 29th, 2024

import cv2
import numpy as np


def ReadVolume(caseImgPaths, caseSegPaths):
  volumeCropped = []

  for i in range(len(caseImgPaths)):
    # Load the images.
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)

    # Extract the ROI.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Crop the ROI.
    x, y, w, h = cv2.boundingRect(roi)
    cropped = roi[y:y + h, x:x + w]

    volumeCropped.append(cropped)

  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])

  for i in range(len(volumeCropped)):
    # Calculate the padding size.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]
    deltaHeight = maxHeight - volumeCropped[i].shape[0]

    # Add padding to the cropped image and place the image in the center.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],  # Image to pad.
      deltaHeight // 2,  # Top padding.
      deltaHeight - deltaHeight // 2,  # Bottom padding.
      deltaWidth // 2,  # Left padding.
      deltaWidth - deltaWidth // 2,  # Right padding.
      cv2.BORDER_CONSTANT,  # Padding type.
      value=0  # Padding value.
    )

    volumeCropped[i] = padded.copy()

  volumeCropped = np.array(volumeCropped)

  return volumeCropped


def CalculateGLCM3DCooccuranceMatrix(volume, d, theta, isNorm=True, ignoreZeros=True):
  N = np.max(volume) + 1
  noOfSlices = volume.shape[0]

  coMatrix = np.zeros((N, N, noOfSlices))

  for xLoc in range(volume.shape[2]):
    for yLoc in range(volume.shape[1]):
      for zLoc in range(volume.shape[0]):

        startLoc = (zLoc, yLoc, xLoc)
        xTarget = xLoc + np.round(d * np.cos(theta) * np.sin(theta))
        yTarget = yLoc - np.round(d * np.sin(theta) * np.sin(theta))
        zTarget = zLoc + np.round(d * np.cos(theta))
        endLoc = (int(zTarget), int(yTarget), int(xTarget))

        # Check if the target location is within the bounds of the matrix.
        if ((endLoc[0] < 0) or (endLoc[0] >= volume.shape[0]) or
          (endLoc[1] < 0) or (endLoc[1] >= volume.shape[1]) or
          (endLoc[2] < 0) or (endLoc[2] >= volume.shape[2])):
          continue

        if (ignoreZeros):
          # Skip the calculation if the pixel values are zero.
          if (volume[endLoc] == 0) or (volume[startLoc] == 0):
            continue

        # Increment the co-occurrence matrix.
        coMatrix[volume[endLoc], volume[startLoc]] += 1

  if (isNorm):
    # Normalize the co-occurrence matrix.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix


def CalculateGLCMFeatures3D(coMatrix):
  d, h, w = coMatrix.shape

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)

  contrast = 0.0
  homogeneity = 0.0
  entropy = 0.0
  dissimilarity = 0.0
  meanX = 0.0
  meanY = 0.0
  meanZ = 0.0

  for i in range(d):
    for j in range(h):
      for k in range(w):
        # Calculate the contrast in the direction of theta.
        contrast += (i - j) ** 2 * coMatrix[i, j, k]

        # Calculate the homogeneity of the co-occurrence matrix.
        homogeneity += coMatrix[i, j, k] / (1 + (i - j) ** 2)

        # Calculate the entropy of the co-occurrence matrix.
        if (coMatrix[i, j, k] > 0):
          entropy -= coMatrix[i, j, k] * np.log(coMatrix[i, j, k])

        # Calculate the mean of the co-occurrence matrix.
        meanX += i * coMatrix[i, j, k]
        meanY += j * coMatrix[i, j, k]
        meanZ += k * coMatrix[i, j, k]

        # Calculate the dissimilarity of the co-occurrence matrix.
        dissimilarity += np.abs(i - j) * coMatrix[i, j, k]

  # Calculate the correlation of the co-occurrence matrix.
  stdDevX = 0.0
  stdDevY = 0.0
  stdDevZ = 0.0
  for i in range(d):
    for j in range(h):
      for k in range(w):
        stdDevX += (i - meanX) ** 2 * coMatrix[i, j, k]
        stdDevY += (j - meanY) ** 2 * coMatrix[i, j, k]
        stdDevZ += (k - meanZ) ** 2 * coMatrix[i, j, k]

  correlation = 0.0
  stdDevX = np.sqrt(stdDevX)
  stdDevY = np.sqrt(stdDevY)
  stdDevZ = np.sqrt(stdDevZ)
  for i in range(d):
    for j in range(h):
      for k in range(w):
        correlation += ((i - meanX) * (j - meanY) * (k - meanZ) * coMatrix[i, j, k] /
                        (stdDevX * stdDevY * stdDevZ))

  return {
    "Energy"       : energy,
    "Contrast"     : contrast,
    "Homogeneity"  : homogeneity,
    "Entropy"      : entropy,
    "Correlation"  : correlation,
    "Dissimilarity": dissimilarity,
    "MeanX"        : meanX,
    "MeanY"        : meanY,
    "MeanZ"        : meanZ,
    "StdDevX"      : stdDevX,
    "StdDevY"      : stdDevY,
    "StdDevZ"      : stdDevZ,
  }


caseImgPaths = [
  r"Segmentation Slices/Segmentation Slice 65.bmp",
  r"Segmentation Slices/Segmentation Slice 66.bmp",
  r"Segmentation Slices/Segmentation Slice 67.bmp",
]
caseSegPaths = [
  r"Volume Slices/Volume Slice 65.bmp",
  r"Volume Slices/Volume Slice 66.bmp",
  r"Volume Slices/Volume Slice 67.bmp",
]

d = 1
theta = 0

volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

coMatrix = CalculateGLCM3DCooccuranceMatrix(
  volumeCropped, d, theta,
  isNorm=True, ignoreZeros=True
)

features = CalculateGLCMFeatures3D(coMatrix)

# Print the GLCM features.
for key in features:
  print(key, ":", features[key])
