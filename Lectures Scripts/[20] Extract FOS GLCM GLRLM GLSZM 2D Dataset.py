# Author: Hossam Magdy Balaha
# Date: June 13th, 2024

import os, cv2, sys, tqdm
import numpy as np
import pandas as pd

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 6)


def FirstOrderFeatures(image, isNorm=True, ignoreZeros=True):
  # Calculate the histogram.
  min = int(np.min(image))
  max = int(np.max(image))
  hist2D = []
  for i in range(min, max + 1):
    hist2D.append(np.count_nonzero(image == i))
  hist2D = np.array(hist2D)

  if (ignoreZeros):
    # Ignore the background.
    hist2D = hist2D[1:]
    min += 1

  if (isNorm):
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

  results = {
    "Count"          : count,
    "Mean"           : mean,
    "Variance"       : variance,
    "StandardDev"    : stdDev,
    "Skewness"       : skewness,
    "Kurtosis"       : kurtosis,
    "Excess Kurtosis": exKurtosis,
  }

  return results


def CalculateGLCMCooccuranceMatrix(image, d, theta, isNorm=True, ignoreZeros=True):
  N = np.max(image) + 1

  coMatrix = np.zeros((N, N))

  for xLoc in range(image.shape[1]):
    for yLoc in range(image.shape[0]):
      startLoc = (yLoc, xLoc)
      xTarget = xLoc + np.round(d * np.cos(theta))
      yTarget = yLoc - np.round(d * np.sin(theta))
      endLoc = (int(yTarget), int(xTarget))

      # Check if the target location is within the bounds of the matrix.
      if ((endLoc[0] < 0) or (endLoc[0] >= image.shape[0]) or
        (endLoc[1] < 0) or (endLoc[1] >= image.shape[1])):
        continue

      if (ignoreZeros):
        # Skip the calculation if the pixel values are zero.
        if (image[endLoc] == 0) or (image[startLoc] == 0):
          continue

      # Increment the co-occurrence matrix.
      coMatrix[image[endLoc], image[startLoc]] += 1

  if (isNorm):
    # Normalize the co-occurrence matrix.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix


def CalculateGLCMFeatures(coMatrix):
  N = coMatrix.shape[0]

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)

  contrast = 0.0
  homogeneity = 0.0
  entropy = 0.0
  dissimilarity = 0.0
  meanX = 0.0
  meanY = 0.0

  for i in range(N):
    for j in range(N):
      # Calculate the contrast in the direction of theta.
      contrast += (i - j) ** 2 * coMatrix[i, j]

      # Calculate the homogeneity of the co-occurrence matrix.
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)

      # Calculate the entropy of the co-occurrence matrix.
      if (coMatrix[i, j] > 0):
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])

      # Calculate the dissimilarity of the co-occurrence matrix.
      dissimilarity += np.abs(i - j) * coMatrix[i, j]

      # Calculate the mean of the co-occurrence matrix.
      meanX += i * coMatrix[i, j]
      meanY += j * coMatrix[i, j]

  # Calculate the correlation of the co-occurrence matrix.
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

  return {
    "Energy"       : energy,
    "Contrast"     : contrast,
    "Homogeneity"  : homogeneity,
    "Entropy"      : entropy,
    "Correlation"  : correlation,
    "Dissimilarity": dissimilarity,
    "MeanX"        : meanX,
    "MeanY"        : meanY,
    "StdDevX"      : stdDevX,
    "StdDevY"      : stdDevY,
  }


def CalculateGLRLMRunLengthMatrix(image, theta, isNorm=True, ignoreZeros=True):
  N = np.max(image) + 1
  R = np.max(image.shape)

  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(image.shape)
  dx = int(np.round(np.cos(theta)))
  dy = int(np.round(np.sin(theta)))

  for i in range(image.shape[0]):  # Rows
    for j in range(image.shape[1]):  # Columns
      # Skip if already seen.
      if (seenMatrix[i, j] == 1):
        continue

      seenMatrix[i, j] = 1  # Mark as seen.
      currentPixel = image[i, j]  # Current pixel value.
      d = 1  # Distance.

      while (
        (i + d * dy >= 0) and
        (i + d * dy < image.shape[0]) and
        (j + d * dx >= 0) and
        (j + d * dx < image.shape[1])
      ):
        if (image[i + d * dy, j + d * dx] == currentPixel):
          seenMatrix[int(i + d * dy), int(j + d * dx)] = 1
          d += 1
        else:
          break

      # Ignore zeros if needed.
      if (ignoreZeros and (currentPixel == 0)):
        continue

      rlMatrix[currentPixel, d - 1] += 1

  if (isNorm):
    # Normalize the run-length matrix.
    rlMatrix = rlMatrix / (np.sum(rlMatrix) + 1e-6)

  return rlMatrix


def CalculateGLRLMFeatures(rlMatrix, image):
  N = np.max(image) + 1
  R = np.max(image.shape)

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
  rp = rlN / np.prod(image.shape)

  # Low Gray Level Run Emphasis.
  lgre = np.sum(
    rlMatrix / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  # High Gray Level Run Emphasis.
  hgre = np.sum(
    rlMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / rlN

  return {
    "Short Run Emphasis"          : sre,
    "Long Run Emphasis"           : lre,
    "Gray Level Non-Uniformity"   : gln,
    "Run Length Non-Uniformity"   : rln,
    "Run Percentage"              : rp,
    "Low Gray Level Run Emphasis" : lgre,
    "High Gray Level Run Emphasis": hgre,
  }


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


def CalculateGLSZMSizeZoneMatrix(image, connectivity=4, isNorm=True, ignoreZeros=True):
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

      # Ignore zeros if needed.
      if (ignoreZeros and (pixel == 0)):
        continue

      szMatrix[pixel, len(zone) - 1] += 1

  if (isNorm):
    # Normalize the size-zone matrix.
    szMatrix = szMatrix / (np.sum(szMatrix) + 1e-6)

  return szMatrix, szDict, N, Z


def CalculateGLSZMFeatures(szMatrix, image, N, Z):
  # Adding 1e-10 to avoid division by zero nor log(0).

  szN = np.sum(szMatrix)

  # Small Zone Emphasis.
  sze = np.sum(
    szMatrix / (np.arange(1, Z + 1) ** 2 + 1e-10),
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
  zp = szN / np.prod(image.shape)

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
  zse = -np.sum(
    (np.sum(szMatrix, axis=0) / szN) * np.log(np.sum(szMatrix, axis=0) / szN + 1e-10),
  )

  # Low Gray Level Zone Emphasis.
  lgze = np.sum(
    szMatrix / (np.arange(1, N + 1)[:, None] ** 2 + 1e-10),
  ).sum() / szN

  # High Gray Level Zone Emphasis.
  hgze = np.sum(
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  # Small Zone Low Gray Level Emphasis.
  szlge = np.sum(
    szMatrix / ((np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2) + 1e-10),
  ).sum() / szN

  # Small Zone High Gray Level Emphasis.
  szhge = np.sum(
    szMatrix * (np.arange(1, N + 1)[:, None] ** 2) / (np.arange(1, Z + 1) ** 2 + 1e-10),
  ).sum() / szN

  # Large Zone Low Gray Level Emphasis.
  lzgle = np.sum(
    szMatrix * (np.arange(1, Z + 1) ** 2) / (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  # Large Zone High Gray Level Emphasis.
  lzhge = np.sum(
    szMatrix * (np.arange(1, Z + 1) ** 2) * (np.arange(1, N + 1)[:, None] ** 2),
  ).sum() / szN

  return {
    "Small Zone Emphasis"                : sze,
    "Large Zone Emphasis"                : lze,
    "Gray Level Non-Uniformity"          : gln,
    "Zone Size Non-Uniformity"           : zsn,
    "Zone Percentage"                    : zp,
    "Gray Level Variance"                : glv,
    "Zone Size Variance"                 : zsv,
    "Zone Size Entropy"                  : zse,
    "Low Gray Level Zone Emphasis"       : lgze,
    "High Gray Level Zone Emphasis"      : hgze,
    "Small Zone Low Gray Level Emphasis" : szlge,
    "Small Zone High Gray Level Emphasis": szhge,
    "Large Zone Low Gray Level Emphasis" : lzgle,
    "Large Zone High Gray Level Emphasis": lzhge,
  }


# YOU WILL NEED TO CHANGE THE PATH OF THE DATASET.
datasetPath = r"..\..\Datasets\COVID-19 Radiography Database"

classes = [
  "COVID",
  "Normal",
  "Viral Pneumonia",
  "Lung_Opacity",
]

# 2D Configurations:
inputSize = (224, 224, 3)
connectivity = 8
d = 1
theta = 0
ignoreZeros = True
applyNorm = True

records = []

# Load the dataset.
for cls in classes[::-1]:
  print(f"Processing class: {cls}")
  clsPath = os.path.join(datasetPath, cls)
  files = os.listdir(clsPath + "/images")
  for file in tqdm.tqdm(files[:2500]):  # Process only 500 images from each class.
    caseImgPath = os.path.join(clsPath, "images", file)
    caseSegPath = os.path.join(clsPath, "masks", file)

    # Load the images.
    caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
    caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

    # Resize the images.
    caseImg = cv2.resize(caseImg, inputSize[:2], interpolation=cv2.INTER_CUBIC)
    caseSeg = cv2.resize(caseSeg, inputSize[:2], interpolation=cv2.INTER_CUBIC)
    caseSeg[caseSeg > 0] = 255

    contours = cv2.findContours(caseSeg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours[0]) != 2):
      continue

    # Sort the contours based on the x value in the bounding rectangle.
    # Sorts in order of increasing x value (ascending).
    contours = sorted(contours[0], key=lambda x: cv2.boundingRect(x)[0], reverse=False)
    rightLung, leftLung = contours[0], contours[1]

    leftLungMask = np.zeros_like(caseSeg)
    cv2.fillPoly(leftLungMask, [leftLung], 255)
    leftLungROI = cv2.bitwise_and(caseImg, leftLungMask)

    rightLungMask = np.zeros_like(caseSeg)
    cv2.fillPoly(rightLungMask, [rightLung], 255)
    rightLungROI = cv2.bitwise_and(caseImg, rightLungMask)

    record = {}

    for i, roi in enumerate([rightLungROI, leftLungROI]):
      # Crop the ROI.
      x, y, w, h = cv2.boundingRect(roi)
      cropped = roi[y:y + h, x:x + w]

      # Extract the features.
      firstOrder = FirstOrderFeatures(cropped, isNorm=applyNorm, ignoreZeros=ignoreZeros)

      glcm = CalculateGLCMCooccuranceMatrix(cropped, d, theta, isNorm=applyNorm, ignoreZeros=ignoreZeros)

      glcmFeatures = CalculateGLCMFeatures(glcm)

      rlMatrix = CalculateGLRLMRunLengthMatrix(cropped, theta, isNorm=applyNorm, ignoreZeros=ignoreZeros)

      rlFeatures = CalculateGLRLMFeatures(rlMatrix, cropped)

      szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(
        cropped, connectivity=connectivity, isNorm=applyNorm, ignoreZeros=ignoreZeros
      )

      szFeatures = CalculateGLSZMFeatures(szMatrix, cropped, N, Z)

      record.update({k + f"_{i + 1}": v for k, v in firstOrder.items()})
      record.update({k + f"_{i + 1}": v for k, v in glcmFeatures.items()})
      record.update({k + f"_{i + 1}": v for k, v in rlFeatures.items()})
      record.update({k + f"_{i + 1}": v for k, v in szFeatures.items()})

    record["Class"] = cls
    records.append(record)

# Save the dataset.
df = pd.DataFrame(records)
df.to_csv("COVID-19 Radiography Database 2D (2500 Records).csv", index=False)
print("Dataset saved successfully.")
