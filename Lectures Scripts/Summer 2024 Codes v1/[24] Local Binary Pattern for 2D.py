# Author: Hossam Magdy Balaha
# Date: June 20th, 2024
# Permissions and Citation: Refer to the README file.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def LocalBinaryPattern2D(image, theta=135, isClockwise=False):
  distance = 1
  windowSize = distance * 2 + 1
  centerX = windowSize // 2
  centerY = windowSize // 2

  lbpMatrix = np.zeros(image.shape, dtype=np.uint8)
  kernel = np.zeros((windowSize, windowSize), dtype=np.uint8)

  paddedA = np.pad(image, distance, mode="constant", constant_values=0)

  for i in range(windowSize ** 2 - 1):
    xLoc = int(centerX + np.round(distance * np.cos(np.radians(theta))))
    yLoc = int(centerY - np.round(distance * np.sin(np.radians(theta))))
    kernel[yLoc, xLoc] = 2 ** i
    theta += -45 if isClockwise else 45

  for y in range(1, image.shape[0] + 1):
    for x in range(1, image.shape[1] + 1):
      region = paddedA[y - distance:y + distance + 1, x - distance:x + distance + 1]
      comp = region >= region[centerY, centerX]
      lbpMatrix[y - 1, x - 1] = np.sum(kernel[comp])

  return lbpMatrix


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

isClockwise = False
theta = 135  # Start from the top-left corner.

lbpMatrix = LocalBinaryPattern2D(cropped, theta, isClockwise)

# Calculate the histogram.
min = int(np.min(lbpMatrix))
max = int(np.max(lbpMatrix))
hist2D = []
for i in range(min, max + 1):
  hist2D.append(np.count_nonzero(lbpMatrix == i))
hist2D = np.array(hist2D)

# Calculate the percentiles.
quantiles = [10, 25, 50, 75, 90]
percentiles = np.percentile(hist2D, quantiles)

print("Percentiles:")
print(percentiles)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.imshow(lbpMatrix, cmap="gray")
plt.title("LBP Image")
plt.axis("off")
plt.tight_layout()
plt.show()

plt.figure()
plt.bar(np.arange(min + 1, max), hist2D[1:-1])
plt.title("2D Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
