# Author: Hossam Magdy Balaha
# Date: June 2nd, 2024
# Permissions and Citation: Refer to the README file.

import numpy as np

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
R = np.max(A.shape)
theta = np.radians(theta)

rlMatrix = np.zeros((N, R))
seenMatrix = np.zeros(A.shape)
dx = int(np.round(np.cos(theta)))
dy = int(np.round(np.sin(theta)))

for i in range(A.shape[0]):  # Rows
  for j in range(A.shape[1]):  # Columns
    # Skip if already seen.
    if (seenMatrix[i, j] == 1):
      continue

    seenMatrix[i, j] = 1  # Mark as seen.
    currentPixel = A[i, j]  # Current pixel value.
    d = 1  # Distance.

    while (
      (i + d * dy >= 0) and
      (i + d * dy < A.shape[0]) and
      (j + d * dx >= 0) and
      (j + d * dx < A.shape[1])
    ):
      if (A[i + d * dy, j + d * dx] == currentPixel):
        seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
        d += 1
      else:
        break

    rlMatrix[currentPixel, d - 1] += 1

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
rp = rlN / np.prod(A.shape)

# Low Gray Level Run Emphasis.
lgre = np.sum(
  rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
).sum() / rlN

# High Gray Level Run Emphasis.
hgre = np.sum(
  rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
).sum() / rlN

print("Short Run Emphasis:", sre)
print("Long Run Emphasis:", lre)
print("Gray Level Non-Uniformity:", gln)
print("Run Length Non-Uniformity:", rln)
print("Run Percentage:", rp)
print("Low Gray Level Run Emphasis:", lgre)
print("High Gray Level Run Emphasis:", hgre)
