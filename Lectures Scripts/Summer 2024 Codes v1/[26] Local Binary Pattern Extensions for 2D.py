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

  paddedImage = np.pad(image, distance, mode="constant", constant_values=0)

  for i in range(windowSize ** 2 - 1):
    xLoc = int(centerX + np.round(distance * np.cos(np.radians(theta))))
    yLoc = int(centerY - np.round(distance * np.sin(np.radians(theta))))
    kernel[yLoc, xLoc] = 2 ** i
    theta += -45 if isClockwise else 45

  for y in range(1, image.shape[0] + 1):
    for x in range(1, image.shape[1] + 1):
      region = paddedImage[y - distance:y + distance + 1, x - distance:x + distance + 1]
      comp = region >= region[centerY, centerX]
      lbpMatrix[y - 1, x - 1] = np.sum(kernel[comp])

  return lbpMatrix


def UniformLocalBinaryPattern2D(image, theta=135, isClockwise=False):
  lbpMatrix = LocalBinaryPattern2D(image, theta, isClockwise)
  uniformMatrix = np.zeros(image.shape, dtype=np.uint8)

  for y in range(image.shape[0]):
    for x in range(image.shape[1]):
      binary = np.binary_repr(lbpMatrix[y, x], width=8)
      transitions = 0
      for i in range(1, 8):
        if (binary[i] != binary[i - 1]):
          transitions += 1
      if (transitions <= 2):
        uniformMatrix[y, x] = int(binary, 2)
        # uniformMatrix[y, x] = lbpMatrix[y, x]

  return uniformMatrix


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
lbpMatrixUniform = UniformLocalBinaryPattern2D(cropped, theta, isClockwise)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.subplot(1, 3, 2)
plt.imshow(lbpMatrix, cmap="gray")
plt.title("LBP Image")
plt.axis("off")
plt.tight_layout()
plt.subplot(1, 3, 3)
plt.imshow(lbpMatrixUniform, cmap="gray")
plt.title("Uniform LBP Image")
plt.axis("off")
plt.show()
