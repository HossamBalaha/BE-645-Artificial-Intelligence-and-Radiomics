# Author: Hossam Magdy Balaha
# Date: May 20th, 2024

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Calculate the histogram.
min = int(np.min(cropped))
max = int(np.max(cropped))
hist2D = []
for i in range(min, max + 1):
  hist2D.append(np.count_nonzero(cropped == i))
hist2D = np.array(hist2D)

# Ignore the background.
hist2D = hist2D[1:]
min += 1

# Calculate the count from the histogram.
count = np.sum(hist2D)

# Determine the range.
rng = np.arange(min, max + 1)

# Calculate the sum from the histogram.
sum = np.sum(hist2D * rng)

# Calculate the mean from the histogram.
mean = sum / count

# Calculate the variance from the histogram.
variance = np.sum(hist2D * (rng - mean) ** 2) / count

# Calculate the standard deviation from the histogram.
stdDev = np.sqrt(variance)

# Calculate the skewness from the histogram.
skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)

# Calculate the kurtosis from the histogram.
kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)

# Calculate the excess kurtosis from the histogram.
exKurtosis = kurtosis - 3

# Print the results.
print("Min: ", min)
print("Max: ", max)
print("Range: ", rng)
print("Count: ", count)
print("Sum: ", sum)
print("Mean: ", mean)
print("Variance: ", variance)
print("Standard Deviation: ", stdDev)
print("Skewness: ", skewness)
print("Kurtosis: ", kurtosis)
print("Excess Kurtosis: ", exKurtosis)

# Plot the histogram.
plt.figure()
plt.bar(np.arange(min, max + 1), hist2D)
plt.title("2D Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
