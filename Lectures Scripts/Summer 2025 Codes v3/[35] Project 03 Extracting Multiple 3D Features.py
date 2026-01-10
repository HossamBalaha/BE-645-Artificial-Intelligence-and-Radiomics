'''
========================================================================
      ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
      ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
      ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 13th, 2025
# Last Modification Date: Jul 14th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import tqdm  # For progress bar in loops.
import numpy as np  # For numerical operations.
import pandas as pd  # For data manipulation and analysis.
from medmnist import AdrenalMNIST3D  # For loading the 3D dataset.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Define the different parameters for feature extraction.
extractionParams3D = {
  "FirstOrderFeatures": {
    "turnOn": True,  # Whether to calculate the first order features.
  },
  "GLCM"              : {
    "d"          : [1, 2, 3],  # Distance between voxel pairs.
    "theta"      : [0, 45],  # Angle (in degrees) for the direction of voxel pairs.
    "isSymmetric": False,  # Whether to make the GLCM symmetric.
    "turnOn"     : False,  # Whether to calculate the GLCM features.
  },
  "GLRLM"             : {
    "theta" : [0, 45],  # Angle (in degrees) for the direction of voxel pairs.
    "turnOn": False,  # Whether to calculate the GLRLM features.
  },
  "GLSZM"             : {
    "connectivity": [6],  # Connectivity type.
    "turnOn"      : False,  # Whether to calculate the GLSZM features.
  },
  "Shape"             : {
    "turnOn": False,  # Whether to calculate the shape features.
  },
  "isNorm"            : True,  # Whether to normalize the features.
  "ignoreZeros"       : True,  # Whether to ignore zero-valued pixels.
  "split"             : "train",  # Dataset split to use for feature extraction.
}

# Convert the theta angles from degrees to radians for GLCM and GLRLM.
extractionParams3D["GLCM"]["theta"] = np.radians(extractionParams3D["GLCM"]["theta"])
extractionParams3D["GLRLM"]["theta"] = np.radians(extractionParams3D["GLRLM"]["theta"])

# Extract the feature names that are turned on from the extractionParams3D dictionary.
turnedOnFeatures = [
  key for key, value in extractionParams3D.items()
  if (isinstance(value, dict) and value.get("turnOn", False))
]
# Check if all features are turned off.
if (len(turnedOnFeatures) == 0):
  raise ValueError("No features are turned on for extraction. Please enable at least one feature.")
# Convert the feature names to a string format suitable for the file name.
turnedOnFeaturesStr = "-".join(turnedOnFeatures)

# Extract the dataset details for the specified dataset.
dataDetails = MedMnistDatasetDetails(AdrenalMNIST3D, split=extractionParams3D["split"])
print("Dataset Details:")
for key, value in dataDetails.items():
  print(f"{key}: {value}")

# Initialize the dataset iterator for the specified split.
dataIterator = MedMnistLoaderIterator(AdrenalMNIST3D, split=extractionParams3D["split"])

# Define the path to save the extracted features.
splitCap = extractionParams3D["split"].capitalize()  # Capitalize the split name for the file path.
featuresPath = rf"Data/{dataDetails['python_class']} ({turnedOnFeaturesStr}) {splitCap} Features.csv"

# Create a list to store the features extracted from each volume.
history = []

# Iterate through each class to process images.
for record in tqdm.tqdm(
  dataIterator,
  desc="Processing files.",
  total=dataDetails["n_samples"][extractionParams3D["split"]]
):
  volume, category, i = record  # Unpack the record into volume, category, and index.

  imageFeatures = {}  # Dictionary to store features for the current volume.

  if (extractionParams3D["FirstOrderFeatures"]["turnOn"]):
    fofList = []
    for j in range(volume.shape[0]):
      fof, hist2D = FirstOrderFeatures2DV2(
        volume[j],  # Extract features from each slice of the volume.
        isNorm=extractionParams3D["isNorm"],  # Whether to normalize the features.
        ignoreZeros=extractionParams3D["ignoreZeros"],  # Whether to ignore zero-valued pixels.
      )
      fofList.append(fof)  # Append the features for each slice.
    fofDF = pd.DataFrame(fofList)  # Convert the list of features to a DataFrame.
    fofAvg = fofDF.mean(axis=0)  # Calculate the average features across all slices.
    # Convert the average features to a dictionary.
    firstOrderFeatures = fofAvg.to_dict()  # Convert the average features to a dictionary.
    # Store the first order features in the imageFeatures dictionary.
    imageFeatures.update(firstOrderFeatures)  # Update the dictionary with first order features.

  if (extractionParams3D["GLCM"]["turnOn"]):
    # If GLCM features are enabled, we will calculate them.
    for d in extractionParams3D["GLCM"]["d"]:
      for theta in extractionParams3D["GLCM"]["theta"]:
        # Calculate the GLCM co-occurrence matrix.
        coMatrix = CalculateGLCMCooccuranceMatrix3D(
          volume,  # The cropped ROI volume.
          d,  # Distance between voxel pairs.
          theta,  # Angle (in radians) for the direction of voxel pairs.
          isSymmetric=extractionParams3D["GLCM"]["isSymmetric"],  # Whether to make the GLCM symmetric.
          isNorm=extractionParams3D["isNorm"],  # Whether to normalize the features.
          ignoreZeros=extractionParams3D["ignoreZeros"],  # Whether to ignore zero-valued pixels.
        )
        # Calculate GLCM features from the co-occurrence matrix.
        glcmFeatures = CalculateGLCMFeaturesOptimized(coMatrix)
        # Update the imageFeatures dictionary with GLCM features.
        imageFeatures.update({
          f"{key}_D{d}_T{np.degrees(theta)}": value
          for key, value in glcmFeatures.items()
        })

  if (extractionParams3D["GLRLM"]["turnOn"]):
    # If GLRLM features are enabled, we will calculate them.
    for theta in extractionParams3D["GLRLM"]["theta"]:
      # Calculate the GLRLM matrix.
      rlMatrix = CalculateGLRLMRunLengthMatrix3D(
        volume,  # The cropped ROI volume.
        theta,  # Angle (in radians) for the direction of voxel pairs.
        isNorm=extractionParams3D["isNorm"],  # Whether to normalize the features.
        ignoreZeros=extractionParams3D["ignoreZeros"],  # Whether to ignore zero-valued pixels.
      )
      # Calculate the GLRLM features.
      glrlmFeatures = CalculateGLRLMFeatures(
        rlMatrix,  # The run-length matrix.
        volume,  # The cropped ROI volume.
      )
      # Update the imageFeatures dictionary with GLRLM features.
      imageFeatures.update({
        f"{key}_T{np.degrees(theta)}": value
        for key, value in glrlmFeatures.items()
      })

  if (extractionParams3D["GLSZM"]["turnOn"]):
    # If GLSZM features are enabled, we will calculate them.
    for connectivity in extractionParams3D["GLSZM"]["connectivity"]:
      # Calculate the GLSZM matrix.
      szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix3D(
        volume,  # The cropped ROI volume.
        connectivity=connectivity,  # Connectivity type (6 or 26).
        isNorm=extractionParams3D["isNorm"],  # Whether to normalize the features.
        ignoreZeros=extractionParams3D["ignoreZeros"],  # Whether to ignore zero-valued pixels.
      )
      # Calculate the GLSZM features.
      glszmFeatures = CalculateGLSZMFeatures(
        szMatrix,  # The size zone matrix.
        volume,  # The cropped ROI volume.
        N,
        Z,
      )
      # Update the imageFeatures dictionary with GLSZM features.
      imageFeatures.update({
        f"{key}_C{connectivity}": value
        for key, value in glszmFeatures.items()
      })

  if (extractionParams3D["Shape"]["turnOn"]):
    # If shape features are enabled, we will calculate them.
    shapeFeatures = ShapeFeatures3D(
      volume,  # The cropped ROI volume.
    )
    # Update the imageFeatures dictionary with shape features.
    imageFeatures.update(shapeFeatures)

  # Append the features to the history list.
  history.append({
    "Index": i,  # Store the index of the volume (as we don't have the filename).
    **imageFeatures,  # Unpack the volume features into the dictionary.
    "Class": category,  # Store the class name.
  })

# Convert the history list to a DataFrame.
featuresDF = pd.DataFrame(history)
# Save the features DataFrame to a CSV file.
featuresDF.to_csv(
  featuresPath,  # Path to save the CSV file.
  index=False,  # Do not include the index in the CSV file.
)

print("Feature extraction completed successfully.")
