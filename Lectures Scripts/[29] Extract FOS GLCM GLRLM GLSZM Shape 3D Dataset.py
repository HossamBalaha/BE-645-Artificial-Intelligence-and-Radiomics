# Author: Hossam Magdy Balaha
# Date: June 28th, 2024
# Permissions and Citation: Refer to the README file.

import os, cv2, sys, tqdm, trimesh
import numpy as np
import pandas as pd

# To avoid RecursionError in large images.
# Default recursion limit is 1000.
sys.setrecursionlimit(10 ** 6)


def ReadVolume(caseImgPaths, caseSegPaths):
  volumeCropped = []

  for i in range(len(caseImgPaths)):
    # Load the images.
    caseImg = cv2.imread(caseImgPaths[i], cv2.IMREAD_GRAYSCALE)
    caseSeg = cv2.imread(caseSegPaths[i], cv2.IMREAD_GRAYSCALE)

    # Extract the ROI.
    roi = cv2.bitwise_and(caseImg, caseSeg)

    # Crop the ROI.
    x, y, w, h = cv2.boundingRect(roi)
    cropped = roi[y:y + h, x:x + w]

    volumeCropped.append(cropped)

  maxWidth = np.max([cropped.shape[1] for cropped in volumeCropped])
  maxHeight = np.max([cropped.shape[0] for cropped in volumeCropped])

  for i in range(len(volumeCropped)):
    # Calculate the padding size.
    deltaWidth = maxWidth - volumeCropped[i].shape[1]
    deltaHeight = maxHeight - volumeCropped[i].shape[0]

    # Add padding to the cropped image and place the image in the center.
    padded = cv2.copyMakeBorder(
      volumeCropped[i],  # Image to pad.
      deltaHeight // 2,  # Top padding.
      deltaHeight - deltaHeight // 2,  # Bottom padding.
      deltaWidth // 2,  # Left padding.
      deltaWidth - deltaWidth // 2,  # Right padding.
      cv2.BORDER_CONSTANT,  # Padding type.
      value=0  # Padding value.
    )

    volumeCropped[i] = padded.copy()

  volumeCropped = np.array(volumeCropped)

  return volumeCropped


def FirstOrderFeatures(volume, isNorm=True, ignoreZeros=True):
  feautres = []

  for image in volume:
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

    feautres.append([count, mean, variance, stdDev, skewness, kurtosis, exKurtosis])

  meanFeatures = np.mean(feautres, axis=0)

  results = {
    "Count"      : meanFeatures[0],
    "Mean"       : meanFeatures[1],
    "Variance"   : meanFeatures[2],
    "StandardDev": meanFeatures[3],
    "Skewness"   : meanFeatures[4],
    "Kurtosis"   : meanFeatures[5],
    "ExKurtosis" : meanFeatures[6]
  }

  return results


def CalculateGLCM3DCooccuranceMatrix(volume, d, theta, isNorm=True, ignoreZeros=True):
  N = np.max(volume) + 1
  noOfSlices = volume.shape[0]

  coMatrix = np.zeros((N, N, noOfSlices))

  for xLoc in range(volume.shape[2]):
    for yLoc in range(volume.shape[1]):
      for zLoc in range(volume.shape[0]):

        startLoc = (zLoc, yLoc, xLoc)
        xTarget = xLoc + np.round(d * np.cos(theta) * np.sin(theta))
        yTarget = yLoc - np.round(d * np.sin(theta) * np.sin(theta))
        zTarget = zLoc + np.round(d * np.cos(theta))
        endLoc = (int(zTarget), int(yTarget), int(xTarget))

        # Check if the target location is within the bounds of the matrix.
        if ((endLoc[0] < 0) or (endLoc[0] >= volume.shape[0]) or
          (endLoc[1] < 0) or (endLoc[1] >= volume.shape[1]) or
          (endLoc[2] < 0) or (endLoc[2] >= volume.shape[2])):
          continue

        if (ignoreZeros):
          # Skip the calculation if the pixel values are zero.
          if (volume[endLoc] == 0) or (volume[startLoc] == 0):
            continue

        # Increment the co-occurrence matrix.
        coMatrix[volume[endLoc], volume[startLoc]] += 1

  if (isNorm):
    # Normalize the co-occurrence matrix.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix


def CalculateGLCMFeatures3D(coMatrix):
  d, h, w = coMatrix.shape

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)

  contrast = 0.0
  homogeneity = 0.0
  entropy = 0.0
  dissimilarity = 0.0
  meanX = 0.0
  meanY = 0.0
  meanZ = 0.0

  for i in range(d):
    for j in range(h):
      for k in range(w):
        # Calculate the contrast in the direction of theta.
        contrast += (i - j) ** 2 * coMatrix[i, j, k]

        # Calculate the homogeneity of the co-occurrence matrix.
        homogeneity += coMatrix[i, j, k] / (1 + (i - j) ** 2)

        # Calculate the entropy of the co-occurrence matrix.
        if (coMatrix[i, j, k] > 0):
          entropy -= coMatrix[i, j, k] * np.log(coMatrix[i, j, k])

        # Calculate the mean of the co-occurrence matrix.
        meanX += i * coMatrix[i, j, k]
        meanY += j * coMatrix[i, j, k]
        meanZ += k * coMatrix[i, j, k]

        # Calculate the dissimilarity of the co-occurrence matrix.
        dissimilarity += np.abs(i - j) * coMatrix[i, j, k]

  # Calculate the correlation of the co-occurrence matrix.
  stdDevX = 0.0
  stdDevY = 0.0
  stdDevZ = 0.0
  for i in range(d):
    for j in range(h):
      for k in range(w):
        stdDevX += (i - meanX) ** 2 * coMatrix[i, j, k]
        stdDevY += (j - meanY) ** 2 * coMatrix[i, j, k]
        stdDevZ += (k - meanZ) ** 2 * coMatrix[i, j, k]

  correlation = 0.0
  stdDevX = np.sqrt(stdDevX)
  stdDevY = np.sqrt(stdDevY)
  stdDevZ = np.sqrt(stdDevZ)
  for i in range(d):
    for j in range(h):
      for k in range(w):
        correlation += ((i - meanX) * (j - meanY) * (k - meanZ) * coMatrix[i, j, k] /
                        (stdDevX * stdDevY * stdDevZ))

  return {
    "Energy"       : energy,
    "Contrast"     : contrast,
    "Homogeneity"  : homogeneity,
    "Entropy"      : entropy,
    "Correlation"  : correlation,
    "Dissimilarity": dissimilarity,
    "MeanX"        : meanX,
    "MeanY"        : meanY,
    "MeanZ"        : meanZ,
    "StdDevX"      : stdDevX,
    "StdDevY"      : stdDevY,
    "StdDevZ"      : stdDevZ,
  }


def CalculateGLRLM3DRunLengthMatrix(volume, theta, isNorm=True, ignoreZeros=True):
  N = np.max(volume) + 1
  R = np.max(volume.shape)

  rlMatrix = np.zeros((N, R))
  seenMatrix = np.zeros(volume.shape)
  dx = int(np.round(np.cos(theta) * np.sin(theta)))
  dy = int(np.round(np.sin(theta) * np.sin(theta)))
  dz = int(np.round(np.cos(theta)))

  for i in range(volume.shape[0]):  # Z-axis
    for j in range(volume.shape[1]):  # Y-axis
      for k in range(volume.shape[2]):  # X-axis
        # Skip if already seen.
        if (seenMatrix[i, j, k] == 1):
          continue

        seenMatrix[i, j, k] = 1  # Mark as seen.
        currentPixel = volume[i, j, k]  # Current pixel value.
        d = 1  # Distance.

        while (
          (i + d * dz >= 0) and
          (i + d * dz < volume.shape[0]) and
          (j + d * dy >= 0) and
          (j + d * dy < volume.shape[1]) and
          (k + d * dx >= 0) and
          (k + d * dx < volume.shape[2])
        ):
          if (volume[i + d * dz, j + d * dy, k + d * dx] == currentPixel):
            seenMatrix[int(i + d * dz), int(j + d * dy), int(k + d * dx)] = 1
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


def CalculateGLRLMFeatures3D(rlMatrix, volume):
  N = np.max(volume) + 1
  R = np.max(volume.shape)

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
  rp = rlN / np.prod(volume.shape)

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


def FindConnected3DRegions(volume, connectivity=6):
  def FindConnected3DRegionsHelper(i, j, k, currentPixel, region, seenMatrix, connectivity=6):
    if (
      (i < 0) or
      (i >= volume.shape[0]) or
      (j < 0) or
      (j >= volume.shape[1]) or
      (k < 0) or
      (k >= volume.shape[2]) or
      (volume[i, j, k] != currentPixel) or
      ((i, j, k) in region)
    ):
      return

    region.add((i, j, k))
    seenMatrix[i, j, k] = 1

    FindConnected3DRegionsHelper(i - 1, j, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j - 1, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j, k - 1, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i + 1, j, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j + 1, k, currentPixel, region, seenMatrix, connectivity)
    FindConnected3DRegionsHelper(i, j, k + 1, currentPixel, region, seenMatrix, connectivity)

    if (connectivity == 26):
      FindConnected3DRegionsHelper(i - 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j - 1, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i - 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j - 1, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j - 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j + 1, k - 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j + 1, k, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i + 1, j, k + 1, currentPixel, region, seenMatrix, connectivity)
      FindConnected3DRegionsHelper(i, j + 1, k + 1, currentPixel, region, seenMatrix, connectivity)

  seenMatrix = np.zeros(volume.shape)

  regions = {}
  for i in range(volume.shape[0]):
    for j in range(volume.shape[1]):
      for k in range(volume.shape[2]):
        if (seenMatrix[i, j, k]):
          continue

        currentPixel = volume[i, j, k]

        if (currentPixel not in regions):
          regions[currentPixel] = []

        region = set()
        FindConnected3DRegionsHelper(i, j, k, currentPixel, region, seenMatrix, connectivity)

        if (len(region) > 0):
          regions[currentPixel].append(region)

  return regions


def CalculateGLSZM3DSizeZoneMatrix3D(volume, connectivity=6, isNorm=True, ignoreZeros=True):
  szDict = FindConnected3DRegions(volume, connectivity=connectivity)

  N = np.max(volume) + 1
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


def CalculateGLSZMFeatures(szMatrix, volume, N, Z):
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
  zp = szN / np.prod(volume.shape)

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


def ShapeFeatures3D(volume):
  # Converts an (n, m, p) matrix into a mesh, using marching_cubes
  # marching_cubes => from skimage import measure
  mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volumeCropped)

  # 1. Volume.
  volume = np.sum(volume)

  # 2. Surface Area.
  surfaceArea = mesh.area

  # 3. Surface to Volume Ratio.
  surfaceToVolumeRatio = surfaceArea / volume

  # 4. Compactness.
  compactness = (volume ** (2 / 3)) / (6 * np.sqrt(np.pi) * surfaceArea)

  # 5. Sphericity.
  sphericity = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surfaceArea

  # Bounding Box.
  bbox = mesh.bounding_box.bounds
  Lmax = np.max(bbox[1] - bbox[0])  # Maximum length of the bounding box.
  Lmin = np.min(bbox[1] - bbox[0])  # Minimum length of the bounding box.
  Lint = np.median(bbox[1] - bbox[0])  # Intermediate length of the bounding box.

  # 6. Elongation.
  elongation = Lmax / Lmin

  # 7. Flatness.
  flatness = Lmin / Lint

  # 8. Rectangularity.
  bboxVolume = np.prod(bbox[1] - bbox[0])  # Volume of the bounding box.
  rectangularity = volume / bboxVolume

  # 9. Spherical Disproportion.
  sphericalDisproportion = (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / surfaceArea

  # 10. Euler Number.
  eulerNumber = mesh.euler_number

  return {
    "Volume"                 : volume,
    "Surface Area"           : surfaceArea,
    "Surface to Volume Ratio": surfaceToVolumeRatio,
    "Compactness"            : compactness,
    "Sphericity"             : sphericity,
    "Elongation"             : elongation,
    "Flatness"               : flatness,
    "Rectangularity"         : rectangularity,
    "Spherical Disproportion": sphericalDisproportion,
    "Euler Number"           : eulerNumber
  }
