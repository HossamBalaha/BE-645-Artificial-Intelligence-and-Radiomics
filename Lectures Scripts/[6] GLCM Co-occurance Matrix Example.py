# Author: Hossam Magdy Balaha
# Date: May 29th, 2024

import numpy as np
import matplotlib.pyplot as plt

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

print("Matrix:")
print(A)

N = np.max(A) + 1

coMatrix = np.zeros((N, N))

for xLoc in range(A.shape[1]):
  for yLoc in range(A.shape[0]):
    startLoc = (yLoc, xLoc)
    xTarget = xLoc + d * np.cos(theta)
    yTarget = yLoc - d * np.sin(theta)
    endLoc = (int(yTarget), int(xTarget))

    # Check if the target location is within the bounds of the matrix.
    if ((endLoc[0] < 0) or (endLoc[0] >= A.shape[0]) or
      (endLoc[1] < 0) or (endLoc[1] >= A.shape[1])):
      continue

    # Increment the co-occurrence matrix.
    coMatrix[A[endLoc], A[startLoc]] += 1

    print(
      f"Start: {startLoc}, End: {endLoc}",
      f"Increment (x={A[startLoc]}, y={A[endLoc]}) by 1."
    )

print("Co-occurrence Matrix:")
print(coMatrix)

# Display the co-occurrence matrix.
plt.figure()
plt.imshow(coMatrix, cmap="gray")
plt.title("Co-occurrence Matrix")
plt.colorbar()
plt.axis("off")
plt.tight_layout()
plt.show()
