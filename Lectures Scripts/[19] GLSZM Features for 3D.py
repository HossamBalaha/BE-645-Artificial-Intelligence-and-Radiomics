# Author: Hossam Magdy Balaha
# Date: June 13th, 2024

import cv2, sys
import numpy as np

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 6)


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


def FindConnected3DRegions(volume, connectivity=6):
  def FindConnected3DRegionsHelper(i, j, k, currentPixel, region, seenMatrix, connectivity=6):
    if (
      (i < 0) or
      (i >= volume.shape[0]) or
      (j < 0) or
      (j >= volume.shape[1]) or
      (k < 0) or
      (k >= volume.shape[2]) or
      (volume[i, j, k] != currentPixel) or
      ((i, j, k) in region)
    ):
      return

    region.add((i, j, k))
    seenMatrix[i, j, k] = 1

    FindConnected3DRegionsHelper(i - 1, j, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j - 1, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j, k - 1, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i + 1, j, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j + 1, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j, k + 1, currentPixel, region, seenMatrix, connectivity)

    if (connectivity == 26):
      FindConnected3DRegionsHelper(i - 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)

  seenMatrix = np.zeros(volume.shape)

  regions = {}
  for i in range(volume.shape[0]):
    for j in range(volume.shape[1]):
      for k in range(volume.shape[2]):
        if (seenMatrix[i, j, k]):
          continue

        currentPixel = volume[i, j, k]

        if (currentPixel not in regions):
          regions[currentPixel] = []

        region = set()
        FindConnected3DRegionsHelper(i, j, k, currentPixel, region, seenMatrix, connectivity)

        if (len(region) > 0):
          regions[currentPixel].append(region)

  return regions


def CalculateGLSZM3DSizeZoneMatrix3D(volume, connectivity=6, isNorm=True, ignoreZeros=True):
  szDict = FindConnected3DRegions(volume, connectivity=connectivity)

  N = np.max(volume) + 1
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


def CalculateGLSZMFeatures(szMatrix, volume, N, Z):
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
  zp = szN / np.prod(volume.shape)

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

volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

connectivity = 6

szMatrix, szDict, N, Z = CalculateGLSZM3DSizeZoneMatrix3D(volumeCropped, connectivity=connectivity)
features = CalculateGLSZMFeatures(szMatrix, volumeCropped, N, Z)

# Print the GLSZM features.
for key in features:
  print(key, ":", features[key])
