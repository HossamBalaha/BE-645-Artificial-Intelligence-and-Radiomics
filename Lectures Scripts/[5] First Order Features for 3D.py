'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 20th, 2024
# Last Modification Date: Jan 14th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import os  # For file and directory operations.
import numpy as np  # For numerical operations.
import pandas as pd  # For data manipulation and saving results to CSV.


def FirstOrderFeatures(img, mask):
  """
  Calculate first-order statistical features from an image using a mask.

  Args:
      img (numpy.ndarray): The input image as a 2D NumPy array.
      mask (numpy.ndarray): The binary mask as a 2D NumPy array.

  Returns:
      results (dict): A dictionary containing the calculated first-order features.
  """
  # Extract the Region of Interest (ROI) using the mask.
  roi = cv2.bitwise_and(img, mask)  # Apply bitwise AND operation to extract the ROI.

  # Crop the ROI to remove unnecessary background.
  x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
  cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

  # Calculate the histogram of the cropped ROI.
  min = int(np.min(cropped))  # Find the minimum pixel value in the cropped ROI.
  max = int(np.max(cropped))  # Find the maximum pixel value in the cropped ROI.
  hist2D = []  # Initialize an empty list to store the histogram values.

  # Loop through each possible value in the range [min, max].
  for i in range(min, max + 1):
    hist2D.append(np.count_nonzero(cropped == i))  # Count occurrences of the value `i` in the cropped ROI.
  hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

  # Ignore the background (assumed to be the first bin in the histogram).
  hist2D = hist2D[1:]  # Remove the first bin (background).
  min += 1  # Adjust the minimum value to exclude the background.

  # Calculate the total count of values in the histogram before normalization.
  freqCount = np.sum(hist2D)  # Sum all frequencies in the histogram.

  # Normalize the histogram to represent probabilities.
  hist2D = hist2D / np.sum(hist2D)  # Divide each bin by the total count to normalize.

  # Calculate the total count of values from the histogram after normalization.
  count = np.sum(hist2D)  # Sum all probabilities in the normalized histogram.

  # Determine the range of values in the histogram.
  rng = np.arange(min, max + 1)  # Create an array of values from `min` to `max`.

  # Calculate the sum of values from the histogram.
  sum = np.sum(hist2D * rng)  # Multiply each value by its frequency and sum the results.

  # Calculate the mean (average) value from the histogram.
  mean = sum / count  # Divide the total sum by the total count.

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
    "Min"            : min,  # Minimum pixel value.
    "Max"            : max,  # Maximum pixel value.
    "Count"          : count,  # Total count of pixels after normalization.
    "Frequency Count": freqCount,  # Total count of pixels before normalization.
    "Sum"            : sum,  # Sum of pixel values.
    "Mean"           : mean,  # Mean pixel value.
    "Variance"       : variance,  # Variance of pixel values.
    "StandardDev"    : stdDev,  # Standard deviation of pixel values.
    "Skewness"       : skewness,  # Skewness of pixel values.
    "Kurtosis"       : kurtosis,  # Kurtosis of pixel values.
    "Excess Kurtosis": exKurtosis,  # Excess kurtosis of pixel values.
  }

  return results


# Define the paths to the volume slices and segmentation masks.
caseVolPath = r"Data/Volume Slices"  # Path to the folder containing volume slices.
caseMskPath = r"Data/Segmentation Slices"  # Path to the folder containing segmentation masks.

# Initialize a list to store the summary of results.
summary = []

# Get the list of volume slice files.
volFiles = os.listdir(caseVolPath)

# Loop through each volume slice file.
for i in range(len(volFiles)):
  # Construct the paths to the volume slice and corresponding segmentation mask.
  caseImgPath = os.path.join(caseVolPath, volFiles[i])  # Path to the volume slice.
  caseSegPath = os.path.join(
    caseMskPath, volFiles[i].replace("Volume", "Segmentation")  # Path to the segmentation mask.
  )

  # Load the volume slice and segmentation mask in grayscale mode.
  caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the volume slice.
  caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

  # Skip images where the segmentation mask is empty (background-only).
  if (np.sum(caseSeg) == 0):
    continue

  # Calculate first-order features for the current volume slice.
  results = FirstOrderFeatures(caseImg, caseSeg)
  results["Image"] = volFiles[i]  # Add the image filename to the results.

  # Append the results to the summary list.
  summary.append(results)

# Save the results to a CSV file.
df = pd.DataFrame(summary)  # Convert the summary list to a Pandas DataFrame.
df.to_csv(
  caseVolPath + " FOF.csv",  # Save the DataFrame to a CSV file.
  index=False,  # Do not include row indices in the CSV file.
)

# Print the results.
print("No. of Images: ", len(summary))  # Print the number of processed images.
# Calculate the mean of each feature across all images.
for key in summary[0].keys():
  # Skip the "Image" key.
  if (key == "Image"):
    continue
  # Extract the values of the current feature from the summary list.
  values = [summary[i][key] for i in range(len(summary))]
  # Print the mean value of the current feature.
  print(key + ": ", np.mean(values))
