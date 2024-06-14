# Author: Hossam Magdy Balaha
# Date: June 12th, 2024

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


A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]

A = np.array(A)

szDict = FindConnectedRegions(A, connectivity=4)

N = np.max(A) + 1
Z = np.max([
  len(zone)
  for zones in szDict.values()
  for zone in zones
])

szMatrix = np.zeros((N, Z))

for pixel, zones in szDict.items():
  for zone in zones:
    szMatrix[pixel, len(zone) - 1] += 1

print("Matrix:")
print(A)
print("Size-Zone Dictionary:")
print(szDict)
print("Size-Zone Matrix:")
print(szMatrix)

# Other Examples to Try:
# A = [
#   [4, 1, 4, 2, 4],
#   [4, 4, 4, 4, 2],
#   [4, 4, 4, 2, 4],
#   [4, 3, 4, 3, 2],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 3, 4, 4],
#   [3, 2, 2, 2],
#   [4, 1, 4, 1],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 1, 2, 4],
#   [3, 3, 3, 2],
#   [4, 1, 4, 1],
# ]
# A = [
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
#   [1, 2, 3, 4],
# ]
# A = [
#   [4, 1, 2, 4],
#   [4, 4, 4, 4],
#   [4, 4, 4, 4],
#   [4, 3, 3, 4],
# ]
