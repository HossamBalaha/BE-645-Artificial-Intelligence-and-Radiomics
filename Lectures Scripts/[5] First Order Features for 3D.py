# Author: Hossam Magdy Balaha
# Date: May 20th, 2024

import cv2, os
import numpy as np


def FirstOrderFeatures(img, mask):
  # Extract the ROI.
  roi = cv2.bitwise_and(img, mask)

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

  # Normalize the histogram.
  hist2D = hist2D / np.sum(hist2D)

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

  results = [
    min, max, count, sum, mean, variance,
    stdDev, skewness, kurtosis, exKurtosis
  ]

  return results


caseVolPath = r"Volume Slices"
caseMskPath = r"Segmentation Slices"

summary = []

volFiles = os.listdir(caseVolPath)
for i in range(len(volFiles)):
  caseImgPath = os.path.join(caseVolPath, volFiles[i])
  caseSegPath = os.path.join(
    caseMskPath, volFiles[i].replace("Volume", "Segmentation")
  )

  caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
  caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

  # Skip background-only images.
  if (np.sum(caseSeg) == 0):
    continue

  results = FirstOrderFeatures(caseImg, caseSeg)
  summary.append(results)

summary = np.array(summary)
meanSummary = np.mean(summary, axis=0)

# Print the results.
print("No. of Images: ", summary.shape[0])
print("Min: ", meanSummary[0])
print("Max: ", meanSummary[1])
print("Count: ", meanSummary[2])
print("Sum: ", meanSummary[3])
print("Mean: ", meanSummary[4])
print("Variance: ", meanSummary[5])
print("Standard Deviation: ", meanSummary[6])
print("Skewness: ", meanSummary[7])
print("Kurtosis: ", meanSummary[8])
print("Excess Kurtosis: ", meanSummary[9])
