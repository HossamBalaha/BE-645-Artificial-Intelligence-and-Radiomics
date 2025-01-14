# Author: Hossam Magdy Balaha
# Date: June 13th, 2024
# Permissions and Citation: Refer to the README file.

import cv2, sys
import numpy as np

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 6)


def FindConnectedRegions(image, connectivity=4):
  def FindConnectedRegionsHelper(i, j, currentPixel, region, seenMatrix, connectivity=4):
    if (
      (i < 0) or
      (i >= image.shape[0]) or
      (j < 0) or
      (j >= image.shape[1]) or
      (image[i, j] != currentPixel) or
      ((i, j) in region)
    ):
      return

    region.add((i, j))
    seenMatrix[i, j] = 1

    FindConnectedRegionsHelper(i - 1, j, currentPixel, region, seenMatrix, connectivity)
    FindConnectedRegionsHelper(i, j - 1, currentPixel, region, seenMatrix, connectivity)
    FindConnectedRegionsHelper(i + 1, j, currentPixel, region, seenMatrix, connectivity)
    FindConnectedRegionsHelper(i, j + 1, currentPixel, region, seenMatrix, connectivity)

    if (connectivity == 8):
      FindConnectedRegionsHelper(i - 1, j - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnectedRegionsHelper(i - 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnectedRegionsHelper(i + 1, j + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnectedRegionsHelper(i + 1, j - 1, currentPixel, region, seenMatrix, connectivity)

  seenMatrix = np.zeros(image.shape)

  regions = {}
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if (seenMatrix[i, j]):
        continue

      currentPixel = image[i, j]

      if (currentPixel not in regions):
        regions[currentPixel] = []

      region = set()
      FindConnectedRegionsHelper(i, j, currentPixel, region, seenMatrix, connectivity)

      if (len(region) > 0):
        regions[currentPixel].append(region)

  return regions


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4, isNorm=True, ignoreZeros=True):
  szDict = FindConnectedRegions(image, connectivity=connectivity)

  N = np.max(image) + 1
  Z = np.max([
    len(zone)
    for zones in szDict.values()
    for zone in zones
  ])

  szMatrix = np.zeros((N, Z))

  for pixel, zones in szDict.items():
    for zone in zones:

      # Ignore zeros if needed.
      if (ignoreZeros and (pixel == 0)):
        continue

      szMatrix[pixel, len(zone) - 1] += 1

  if (isNorm):
    # Normalize the size-zone matrix.
    szMatrix = szMatrix / (np.sum(szMatrix) + 1e-6)

  return szMatrix, szDict, N, Z


def CalculateGLSZMFeatures(szMatrix, image, N, Z):
  # Adding 1e-10 to avoid division by zero nor log(0).

  szN = np.sum(szMatrix)

  # Small Zone Emphasis.
  sze = np.sum(
    szMatrix / (np.arange(1, Z + 1) ** 2 + 1e-10),
  ).sum() / szN

  # Large Zone Emphasis.
  lze = np.sum(
    szMatrix * (np.arange(1, Z + 1) ** 2),
  ).sum() / szN

  # Gray Level Non-Uniformity.
  gln = np.sum(
    np.sum(szMatrix, axis=1) ** 2,  # Sum of each row.
  ) / szN

  # Zone Size Non-Uniformity.
  zsn = np.sum(
    np.sum(szMatrix, axis=0) ** 2,  # Sum of each column.
  ) / szN

  # Zone Percentage.
  zp = szN / np.prod(image.shape)

  # Gray Level Variance.
  glv = np.sum(
    (np.arange(N) ** 2) * np.sum(szMatrix, axis=1),
  ) - (np.sum(
    (np.arange(N) * np.sum(szMatrix, axis=1)),
  ) ** 2)

  # Zone Size Variance.
  zsv = np.sum(
    (np.arange(Z) ** 2) * np.sum(szMatrix, axis=0),
  ) - (np.sum(
    (np.arange(Z) * np.sum(szMatrix, axis=0)),
  ) ** 2)

  # Zone Size Entropy.
  zse = -np.sum(
    (np.sum(szMatrix, axis=0) / szN) * np.log(np.sum(szMatrix, axis=0) / szN + 1e-10),
  )

  # Low Gray Level Zone Emphasis.
  lgze = np.sum(
    szMatrix / (np.arange(1, N + 1)[:, None] ** 2 + 1e-10),
  ).sum() / szN

  # High Gray Level Zone Emphasis.
  hgze = np.sum(
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  # Small Zone Low Gray Level Emphasis.
  szlge = np.sum(
    szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + 1e-10),
  ).sum() / szN

  # Small Zone High Gray Level Emphasis.
  szhge = np.sum(
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / (np.arange(1, Z + 1) ** 2 + 1e-10),
  ).sum() / szN

  # Large Zone Low Gray Level Emphasis.
  lzgle = np.sum(
    szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  # Large Zone High Gray Level Emphasis.
  lzhge = np.sum(
    szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  return {
    "Small Zone Emphasis"                : sze,
    "Large Zone Emphasis"                : lze,
    "Gray Level Non-Uniformity"          : gln,
    "Zone Size Non-Uniformity"           : zsn,
    "Zone Percentage"                    : zp,
    "Gray Level Variance"                : glv,
    "Zone Size Variance"                 : zsv,
    "Zone Size Entropy"                  : zse,
    "Low Gray Level Zone Emphasis"       : lgze,
    "High Gray Level Zone Emphasis"      : hgze,
    "Small Zone Low Gray Level Emphasis" : szlge,
    "Small Zone High Gray Level Emphasis": szhge,
    "Large Zone Low Gray Level Emphasis" : lzgle,
    "Large Zone High Gray Level Emphasis": lzhge,
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

connectivity = 8

szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(cropped, connectivity=connectivity, isNorm=True, ignoreZeros=True)
features = CalculateGLSZMFeatures(szMatrix, cropped, N, Z)

# Print the GLSZM features.
for key in features:
  print(key, ":", features[key])
