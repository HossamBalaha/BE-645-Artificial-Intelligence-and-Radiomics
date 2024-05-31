# Author: Hossam Magdy Balaha
# Date: May 29th, 2024

import numpy as np

d = 1
theta = 0

A = [
  [1, 2, 3, 2, 1],
  [3, 1, 2, 1, 0],
  [0, 3, 2, 2, 1],
  [1, 1, 1, 2, 2],
  [3, 2, 2, 1, 0],
]
A = np.array(A)

N = np.max(A) + 1

coMatrix = np.zeros((N, N))

for xLoc in range(A.shape[1]):
  for yLoc in range(A.shape[0]):
    startLoc = (yLoc, xLoc)
    xTarget = xLoc + np.round(d * np.cos(theta))
    yTarget = yLoc - np.round(d * np.sin(theta))
    endLoc = (int(yTarget), int(xTarget))

    # Check if the target location is within the bounds of the matrix.
    if ((endLoc[0] < 0) or (endLoc[0] >= A.shape[0]) or
      (endLoc[1] < 0) or (endLoc[1] >= A.shape[1])):
      continue

    # Increment the co-occurrence matrix.
    coMatrix[A[endLoc], A[startLoc]] += 1

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

# Print the results.
print("Energy: ", energy)
print("Contrast: ", contrast)
print("Homogeneity: ", homogeneity)
print("Entropy: ", entropy)
print("Correlation: ", correlation)
print("Dissimilarity: ", dissimilarity)
