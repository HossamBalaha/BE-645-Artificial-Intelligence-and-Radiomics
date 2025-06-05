'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: May 20th, 2024
# Last Modification Date: Jun 5th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import cv2  # For image processing tasks.
import os  # For file and directory operations.
import numpy as np  # For numerical operations.
import pandas as pd  # For data manipulation and saving results to CSV.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

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

  # Check if the files exist.
  if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
    raise FileNotFoundError("One or more files were not found. Please check the file paths.")

  # Load the volume slice and segmentation mask in grayscale mode.
  caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the volume slice.
  caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

  # Skip images where the segmentation mask is empty (background-only).
  if (np.sum(caseSeg) == 0):
    continue

  # Calculate first-order features for the current volume slice.
  results, hist2D = FirstOrderFeatures2D(caseImg, caseSeg)
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
print("First Order Features:")  # Print the header for the features.
print("No. of Images:", len(summary))  # Print the number of processed images.
# Calculate the mean of each feature across all images.
for key in summary[0].keys():
  # Skip the "Image" key.
  if (key == "Image"):
    continue
  # Extract the values of the current feature from the summary list.
  values = [summary[i][key] for i in range(len(summary))]
  # Print the mean value of the current feature.
  print(key + ":", np.round(np.mean(values), 4))
