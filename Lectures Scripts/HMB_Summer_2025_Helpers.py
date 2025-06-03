'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 21st, 2025
# Last Modification Date: June 3rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import numpy as np  # For numerical operations.


def FirstOrderFeatures2D(img, mask, isNorm=True, ignoreZeros=True):
  """
  Calculate first-order statistical features from an image using a mask.

  Args:
      img (numpy.ndarray): The input image as a 2D NumPy array.
      mask (numpy.ndarray): The binary mask as a 2D NumPy array.
      isNorm (bool): Flag to indicate whether to normalize the histogram.
      ignoreZeros (bool): Flag to indicate whether to ignore zeros in the histogram.

  Returns:
      results (dict): A dictionary containing the calculated first-order features.
      hist2D (numpy.ndarray): The histogram of the pixel values in the region of interest.
  """
  # Extract the Region of Interest (ROI) using the mask.
  roi = cv2.bitwise_and(img, mask)  # Apply bitwise AND operation to extract the ROI.

  # Crop the ROI to remove unnecessary background.
  x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
  cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

  # Calculate the histogram of the cropped ROI.
  minVal = int(np.min(cropped))  # Find the minimum pixel value in the cropped ROI.
  maxVal = int(np.max(cropped))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [minVal, maxVal].
  for i in range(minVal, maxVal + 1):
    hist2D.append(np.count_nonzero(cropped == i))  # Count occurrences of the value `i` in the cropped ROI.
  hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

  # If ignoreZeros is True, set the first bin (background) to zero.
  if (ignoreZeros and (minVal == 0)):
    # Ignore the background (assumed to be the first bin in the histogram).
    hist2D = hist2D[1:]  # Remove the first bin (background).
    minVal += 1  # Adjust the minimum value to exclude the background.

  # Calculate the total count of values in the histogram before normalization.
  freqCount = np.sum(hist2D)  # Sum all frequencies in the histogram.

  # Normalize the histogram if the flag is set.
  if (isNorm):
    # Normalize the histogram to represent probabilities.
    hist2D = hist2D / np.sum(hist2D)  # Divide each bin by the total count to normalize.

  # Calculate the total count of values from the histogram after normalization.
  count = np.sum(hist2D)  # Sum all probabilities in the normalized histogram.

  # Determine the range of values in the histogram.
  rng = np.arange(minVal, maxVal + 1)  # Create an array of values from `minVal` to `maxVal`.

  # Calculate the sum of values from the histogram.
  sumVal = np.sum(hist2D * rng)  # Multiply each value by its frequency and sum the results.

  # Calculate the mean (average) value from the histogram.
  mean = sumVal / count  # Divide the total sum by the total count.

  # Calculate the variance from the histogram.
  variance = np.sum(hist2D * (rng - mean) ** 2) / count  # Measure of the spread of the data.

  # Calculate the standard deviation from the histogram.
  stdDev = np.sqrt(variance)  # Square root of the variance.

  # Calculate the skewness from the histogram.
  skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)  # Measure of asymmetry in the data.

  # Calculate the kurtosis from the histogram.
  kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)  # Measure of the "tailedness" of the data.

  # Calculate the excess kurtosis from the histogram.
  exKurtosis = kurtosis - 3  # Excess kurtosis relative to a normal distribution.

  # Store the results in a dictionary.
  results = {
    "Min"               : minVal,  # Minimum pixel value.
    "Max"               : maxVal,  # Maximum pixel value.
    "Count"             : count,  # Total count of pixels after normalization.
    "Frequency Count"   : freqCount,  # Total count of pixels before normalization.
    "Sum"               : sumVal,  # Sum of pixel values.
    "Mean"              : mean,  # Mean pixel value.
    "Variance"          : variance,  # Variance of pixel values.
    "Standard Deviation": stdDev,  # Standard deviation of pixel values.
    "Skewness"          : skewness,  # Skewness of pixel values.
    "Kurtosis"          : kurtosis,  # Kurtosis of pixel values.
    "Excess Kurtosis"   : exKurtosis,  # Excess kurtosis of pixel values.
  }

  return results, hist2D


def CalculateGLCMCooccuranceMatrix(image, d, theta, isSymmetric=False, isNorm=True, ignoreZeros=True):
  """
  Calculate the Gray-Level Co-occurrence Matrix (GLCM) for a given image.

  Args:
      image (numpy.ndarray): The input image as a 2D NumPy array.
      d (int): The distance between pixel pairs.
      theta (float): The angle (in radians) for the direction of pixel pairs.
      isSymmetric (bool): Whether to make the GLCM symmetric. Default is False.
      isNorm (bool): Whether to normalize the GLCM. Default is True.
      ignoreZeros (bool): Whether to ignore zero-valued pixels. Default is True.

  Returns:
      coMatrix (numpy.ndarray): The calculated GLCM.
  """
  # Determine the number of unique intensity levels in the matrix.
  minA = np.min(image)  # Minimum intensity value.
  maxA = np.max(image)  # Maximum intensity value.
  N = maxA - minA + 1  # Number of unique intensity levels.

  if (d < 1):
    raise ValueError("The distance between voxel pairs should be greater than or equal to 1.")
  elif (d >= N):
    raise ValueError("The distance between voxel pairs should be less than the number of unique intensity levels.")

  # Initialize the co-occurrence matrix with zeros.
  coMatrix = np.zeros((N, N))  # Create an N x N matrix filled with zeros.

  # Iterate over each pixel in the image to calculate the GLCM.
  for xLoc in range(image.shape[1]):  # Loop through columns.
    for yLoc in range(image.shape[0]):  # Loop through rows.
      startLoc = (yLoc, xLoc)  # Current pixel location (row, column).

      # Calculate the target pixel location based on distance and angle.
      xTarget = xLoc + np.round(d * np.cos(theta))  # Target column.
      yTarget = yLoc - np.round(d * np.sin(theta))  # Target row.
      endLoc = (int(yTarget), int(xTarget))  # Target pixel location.

      # Check if the target location is within the bounds of the image.
      if (
        (endLoc[0] < 0)  # Target row is above the top edge.
        or (endLoc[0] >= image.shape[0])  # Target row is below the bottom edge.
        or (endLoc[1] < 0)  # Target column is to the left of the left edge.
        or (endLoc[1] >= image.shape[1])  # Target column is to the right of the right edge.
      ):
        continue  # Skip this pair if the target is out of bounds.

      if (ignoreZeros):
        # Skip the calculation if the pixel values are zero.
        if ((image[startLoc] == 0) or (image[endLoc] == 0)):
          continue

      # (- minA) is added to work with matrices that does not start from 0.
      # Increment the count for the pair (start, end).
      # image[startLoc] and image[endLoc] are the intensity values at the start and end locations.
      startPixel = image[startLoc] - minA  # Adjust start pixel value.
      endPixel = image[endLoc] - minA  # Adjust end pixel value.

      # Increment the co-occurrence matrix at the corresponding location.
      coMatrix[endPixel, startPixel] += 1

  # If symmetric, add the transpose of the co-occurrence matrix to itself.
  if (isSymmetric):
    coMatrix += coMatrix.T  # Make the GLCM symmetric.

  # Normalize the co-occurrence matrix if requested.
  if (isNorm):
    # Divide each element by the sum of all elements.
    # 1e-6 is added to avoid division by zero.
    coMatrix = coMatrix / (np.sum(coMatrix) + 1e-6)

  return coMatrix  # Return the calculated GLCM.


def CalculateGLCMFeaturesOptimized(coMatrix):
  """
  Calculate texture features from a Gray-Level Co-occurrence Matrix (GLCM).

  Args:
      coMatrix (numpy.ndarray): The GLCM as a 2D NumPy array.

  Returns:
      features (dict): A dictionary containing the calculated texture features.
  """
  N = coMatrix.shape[0]  # Number of unique intensity levels.

  # Calculate the energy of the co-occurrence matrix.
  energy = np.sum(coMatrix ** 2)  # Sum of the squares of all elements in the GLCM.

  # Initialize variables for texture features.
  contrast = 0.0  # Initialize contrast.
  homogeneity = 0.0  # Initialize homogeneity.
  entropy = 0.0  # Initialize entropy.
  dissimilarity = 0.0  # Initialize dissimilarity.
  meanX = 0.0  # Initialize mean of rows.
  meanY = 0.0  # Initialize mean of columns.

  # Loop through each element in the GLCM to calculate texture features.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      # Calculate the contrast in the direction of theta.
      contrast += (i - j) ** 2 * coMatrix[i, j]  # Weighted sum of squared differences.

      # Calculate the homogeneity of the co-occurrence matrix.
      homogeneity += coMatrix[i, j] / (1 + (i - j) ** 2)  # Weighted sum of inverse differences.

      # Calculate the entropy of the co-occurrence matrix.
      if (coMatrix[i, j] > 0):  # Check if the value is greater than zero.
        entropy -= coMatrix[i, j] * np.log(coMatrix[i, j])  # Sum of -p * log(p).

      # Calculate the dissimilarity of the co-occurrence matrix.
      dissimilarity += np.abs(i - j) * coMatrix[i, j]  # Weighted sum of absolute differences.

      # Calculate the mean of the co-occurrence matrix.
      meanX += i * coMatrix[i, j]  # Weighted sum of row indices.
      meanY += j * coMatrix[i, j]  # Weighted sum of column indices.

  totalSum = np.sum(coMatrix)  # Calculate the sum of all elements in the GLCM.
  meanX /= totalSum  # Calculate mean of rows.
  meanY /= totalSum  # Calculate mean of columns.

  # Calculate the standard deviation of rows and columns.
  stdDevX = 0.0  # Initialize standard deviation of rows.
  stdDevY = 0.0  # Initialize standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      stdDevX += (i - meanX) ** 2 * coMatrix[i, j]  # Weighted sum of squared row differences.
      stdDevY += (j - meanY) ** 2 * coMatrix[i, j]  # Weighted sum of squared column differences.

  # Calculate the correlation of the co-occurrence matrix.
  correlation = 0.0  # Initialize correlation.
  stdDevX = np.sqrt(stdDevX)  # Calculate standard deviation of rows.
  stdDevY = np.sqrt(stdDevY)  # Calculate standard deviation of columns.
  for i in range(N):  # Loop through rows.
    for j in range(N):  # Loop through columns.
      correlation += (
        (i - meanX) * (j - meanY) * coMatrix[i, j] / (stdDevX * stdDevY)
      )  # Weighted sum of normalized differences.

  # Return the calculated features as a dictionary.
  return {
    "Energy"       : energy,  # Energy of the GLCM.
    "Contrast"     : contrast,  # Contrast of the GLCM.
    "Homogeneity"  : homogeneity,  # Homogeneity of the GLCM.
    "Entropy"      : entropy,  # Entropy of the GLCM.
    "Correlation"  : correlation,  # Correlation of the GLCM.
    "Dissimilarity": dissimilarity,  # Dissimilarity of the GLCM.
    "TotalSum"     : totalSum,  # Sum of all elements in the GLCM.
    "MeanX"        : meanX,  # Mean of rows.
    "MeanY"        : meanY,  # Mean of columns.
    "StdDevX"      : stdDevX,  # Standard deviation of rows.
    "StdDevY"      : stdDevY,  # Standard deviation of columns.
  }


def PreprocessBrainTumorDatasetFigshare1512427(
  datasetPath,  # Path to the .mat file containing the image data.
  storagePath,  # Path to save the converted image.
  isResize=False,  # Flag to indicate whether to resize the image.
  newSize=(256, 256),  # New size for resizing the image if isResize is True.
  separateFolders=False,  # Flag to indicate whether to save images and masks in separate folders.
):
  """
  Preprocess the Brain Tumor Dataset from Figshare 1512427.
  Link: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

  Args:
      datasetPath (str): Path to the .mat files containing the images data.
      storagePath (str): Path to save the converted image.
      isResize (bool): Flag to indicate whether to resize the image.
      newSize (tuple): New size for resizing the image if isResize is True.
      separateFolders (bool): Flag to indicate whether to save images and masks in separate folders.

  Returns:
      None
  """
  # Install using pip install hdf5storage.
  import hdf5storage, os, tqdm  # Import necessary libraries.

  # Check if the datasetPath folder exists.
  if (not os.path.exists(datasetPath)):
    raise FileNotFoundError(f"The dataset path '{datasetPath}' does not exist.")

  # List all files in the dataset path.
  files = os.listdir(datasetPath)

  # Filter the files to include only .mat files.
  files = [file for file in files if file.endswith(".mat")]  # Keep only .mat files.

  # Key to access the image data in the loaded dictionary.
  key = "cjdata"

  # Create the storage path if it does not exist.
  os.makedirs(storagePath, exist_ok=True)  # Create the directory if it does not exist.

  # Loop through each file in the dataset.
  for file in tqdm.tqdm(files):
    # Construct the full path to the .mat file.
    filePath = os.path.join(datasetPath, file)

    # Load the image data from the .mat file.
    data = hdf5storage.loadmat(filePath)  # Load the .mat file.

    # Extract the data using the specified key.
    matData = data[key][0]

    # Extract the label (tumor type) from the data.
    label = str(int(matData[0][0].squeeze()))

    # Extract the image data from the loaded dictionary.
    imgData = matData[2]  # The image data is the third element in the array.

    # Extract the mask from the loaded dictionary.
    maskData = matData[4]  # The mask is the fifth element in the array.

    # Convert the image data to a NumPy array.
    imgData = np.array(imgData, dtype=np.float32)  # Convert to float32 for processing.
    # Convert the mask data to a NumPy array.
    maskData = np.array(maskData, dtype=np.float32)  # Convert to float32 for processing.

    # Find the min and max pixel values in the image.
    minImg, maxImg = np.min(imgData), np.max(imgData)
    # Normalize the image data to the range [0, 255].
    imgData = (imgData - minImg) / (maxImg - minImg) * 255.0  # Normalize to [0, 255].
    imgData = imgData.astype(np.uint8)  # Convert to uint8 for image representation.

    # Find the min and max pixel values in the mask.
    minMask, maxMask = np.min(maskData), np.max(maskData)
    # Normalize the mask data to the range [0, 255].
    maskData = (maskData - minMask) / (maxMask - minMask) * 255.0  # Normalize to [0, 255].
    maskData = cv2.threshold(maskData, 0, 255, cv2.THRESH_BINARY)[1]  # Convert to binary mask.
    maskData = maskData.astype(np.uint8)  # Convert to uint8 for binary mask representation.

    # If isResize is True, resize the image and mask to the new size.
    if (isResize):
      imgData = cv2.resize(imgData, newSize, interpolation=cv2.INTER_CUBIC)  # Resize the image.
      maskData = cv2.resize(maskData, newSize, interpolation=cv2.INTER_CUBIC)  # Resize the mask.

    # Get the base name of the file.
    fileName = os.path.basename(file)
    # Get the file name without extension.
    fileNameNoExt = os.path.splitext(fileName)[0]

    if (separateFolders):
      # Construct the full path for the image directory.
      imagesFolder = os.path.join(storagePath, "images", label)  # Create a directory for images.
      # Create the image directory if it does not exist.
      os.makedirs(imagesFolder, exist_ok=True)  # Create the directory for images.

      # Construct the full path for the mask directory.
      masksFolder = os.path.join(storagePath, "masks", label)  # Create a directory for masks.
      # Create the mask directory if it does not exist.
      os.makedirs(masksFolder, exist_ok=True)  # Create the directory for masks.

      # Construct the full path for the image and mask files.
      imagePath = os.path.join(imagesFolder, f"{fileNameNoExt}.png")  # Save the image as PNG.
      maskPath = os.path.join(masksFolder, f"{fileNameNoExt}.png")  # Save the mask as PNG.

    else:
      # Construct the full path for the label directory.
      labelPath = os.path.join(storagePath, label)  # Create a directory for the label.
      # Create the label directory if it does not exist.
      os.makedirs(labelPath, exist_ok=True)  # Create the directory for the label.

      # Construct the full path for the image and mask files.
      imagePath = os.path.join(labelPath, f"{fileNameNoExt}.png")  # Save the image as PNG.
      maskPath = os.path.join(labelPath, f"{fileNameNoExt}_mask.png")  # Save the mask as PNG.

    # Save the image and mask to the specified paths.
    cv2.imwrite(imagePath, imgData)  # Save the image.
    cv2.imwrite(maskPath, maskData)  # Save the mask.
