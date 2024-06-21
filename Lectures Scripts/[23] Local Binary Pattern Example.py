# Author: Hossam Magdy Balaha
# Date: June 20th, 2024
# Permissions and Citation: Refer to the README file.

import numpy as np

A = [
  [17, 32, 32, 47, 11],
  [37, 21, 22, 18, 4],
  [16, 23, 40, 21, 11],
  [13, 55, 41, 28, 12],
  [23, 42, 22, 13, 10],
]
A = np.array(A, dtype=np.uint8)
distance = 1
isClockwise = True
originalTheta = 135  # Start from the top-left corner.

windowSize = distance * 2 + 1
centerX = windowSize // 2
centerY = windowSize // 2

lbpMatrix = np.zeros(A.shape, dtype=np.uint8)
kernel = np.zeros((windowSize, windowSize), dtype=np.uint8)

paddedA = np.pad(A, distance, mode="constant", constant_values=0)

theta = originalTheta
for i in range(windowSize ** 2 - 1):
  xLoc = int(centerX + np.round(distance * np.cos(np.radians(theta))))
  yLoc = int(centerY - np.round(distance * np.sin(np.radians(theta))))
  kernel[yLoc, xLoc] = 2 ** i
  theta += -45 if isClockwise else 45

for y in range(1, A.shape[0] + 1):
  for x in range(1, A.shape[1] + 1):
    region = paddedA[y - distance:y + distance + 1, x - distance:x + distance + 1]
    comp = region >= region[centerY, centerX]
    lbpMatrix[y - 1, x - 1] = np.sum(kernel[comp])
    # lbpMatrix[y - 1, x - 1] = np.sum(kernel * comp)

print("Original Matrix:")
print(A)
print()
print(f"LBP Matrix (Theta={originalTheta}, Clockwise={isClockwise}):")
print(lbpMatrix)
