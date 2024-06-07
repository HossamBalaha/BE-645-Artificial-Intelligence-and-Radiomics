# Author: Hossam Magdy Balaha
# Date: June 6th, 2024

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


def CalculateGLRLM3DRunLengthMatrix(volume, theta, isNorm=True, ignoreZeros=True):
  N = np.max(volume) + 1
  R = np.max(volume.shape)

  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(volume.shape)
  dx = int(np.round(np.cos(theta) * np.sin(theta)))
  dy = int(np.round(np.sin(theta) * np.sin(theta)))
  dz = int(np.round(np.cos(theta)))

  for i in range(volume.shape[0]):  # Z-axis
    for j in range(volume.shape[1]):  # Y-axis
      for k in range(volume.shape[2]):  # X-axis
        # Skip if already seen.
        if (seenMatrix[i, j, k] == 1):
          continue

        seenMatrix[i, j, k] = 1  # Mark as seen.
        currentPixel = volume[i, j, k]  # Current pixel value.
        d = 1  # Distance.

        while (
          (i + d * dz >= 0) and
          (i + d * dz < volume.shape[0]) and
          (j + d * dy >= 0) and
          (j + d * dy < volume.shape[1]) and
          (k + d * dx >= 0) and
          (k + d * dx < volume.shape[2])
        ):
          if (volume[i + d * dz, j + d * dy, k + d * dx] == currentPixel):
            seenMatrix[int(i + d * dz), int(j + d * dy), int(k + d * dx)] = 1
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


def CalculateGLRLMFeatures3D(rlMatrix, volume):
  N = np.max(volume) + 1
  R = np.max(volume.shape)

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
  rp = rlN / np.prod(volume.shape)

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


caseImgPaths = [
  r"Volume Slices/Volume Slice 65.bmp",
  r"Volume Slices/Volume Slice 66.bmp",
  r"Volume Slices/Volume Slice 67.bmp",

]
caseSegPaths = [
  r"Segmentation Slices/Segmentation Slice 65.bmp",
  r"Segmentation Slices/Segmentation Slice 66.bmp",
  r"Segmentation Slices/Segmentation Slice 67.bmp",
]

theta = 0

volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

coMatrix = CalculateGLRLM3DRunLengthMatrix(
  volumeCropped, theta,
  isNorm=True, ignoreZeros=True
)

features = CalculateGLRLMFeatures3D(coMatrix, volumeCropped)

# Print the GLCM features.
for key in features:
  print(key, ":", features[key])
