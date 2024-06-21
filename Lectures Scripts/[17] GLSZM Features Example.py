# Author: Hossam Magdy Balaha
# Date: June 13th, 2024
# Permissions and Citation: Refer to the README file.

import numpy as np


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


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4):
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
      szMatrix[pixel, len(zone) - 1] += 1

  return szMatrix, szDict, N, Z


A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]

A = np.array(A)

szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(A, connectivity=8)

szN = np.sum(szMatrix)

# Small Zone Emphasis.
sze = np.sum(
  szMatrix / (np.arange(1, Z + 1) ** 2),
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
zp = szN / np.prod(A.shape)

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
# Adding 1e-10 to avoid log(0).
zse = -np.sum(
  (np.sum(szMatrix, axis=0) / szN) * np.log(np.sum(szMatrix, axis=0) / szN + 1e-10),
)

# Low Gray Level Zone Emphasis.
lgze = np.sum(
  szMatrix / (np.arange(1, N + 1)[:, None] ** 2),
).sum() / szN

# High Gray Level Zone Emphasis.
hgze = np.sum(
  szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
).sum() / szN

# Small Zone Low Gray Level Emphasis.
# Adding 1e-10 to avoid division by zero.
szlge = np.sum(
  szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + 1e-10),
).sum() / szN

# Small Zone High Gray Level Emphasis.
szhge = np.sum(
  szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / (np.arange(1, Z + 1) ** 2),
).sum() / szN

# Large Zone Low Gray Level Emphasis.
lzgle = np.sum(
  szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
).sum() / szN

# Large Zone High Gray Level Emphasis.
lzhge = np.sum(
  szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
).sum() / szN

print("Small Zone Emphasis:", sze)
print("Large Zone Emphasis:", lze)
print("Gray Level Non-Uniformity:", gln)
print("Zone Size Non-Uniformity:", zsn)
print("Zone Percentage:", zp)
print("Gray Level Variance:", glv)
print("Zone Size Variance:", zsv)
print("Zone Size Entropy:", zse)
print("Low Gray Level Zone Emphasis:", lgze)
print("High Gray Level Zone Emphasis:", hgze)
print("Small Zone Low Gray Level Emphasis:", szlge)
print("Small Zone High Gray Level Emphasis:", szhge)
print("Large Zone Low Gray Level Emphasis:", lzgle)
print("Large Zone High Gray Level Emphasis:", lzhge)
