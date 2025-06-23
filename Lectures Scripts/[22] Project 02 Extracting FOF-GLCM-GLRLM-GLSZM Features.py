'''
========================================================================
      ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
      ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
      ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 19th, 2025
# Last Modification Date: Jun 23rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.
import cv2  # For image processing tasks.
import tqdm  # For progress bar in loops.
import numpy as np  # For numerical operations.
import pandas as pd  # For data manipulation and analysis.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Define the different parameters for feature extraction.
extractionParams = {
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
    "theta" : [0],  # Angle (in degrees) for the direction of voxel pairs.
    "turnOn": False,  # Whether to calculate the GLRLM features.
  },
  "GLSZM"             : {
    "connectivity": [4],  # Connectivity type.
    "turnOn"      : False,  # Whether to calculate the GLSZM features.
  },
  "targetSize"        : (128, 128),  # Target size for resizing images.
  "isNorm"            : True,  # Whether to normalize the features.
  "ignoreZeros"       : True,  # Whether to ignore zero-valued pixels.
  "maxRegions"        : 2,  # Maximum number of regions to extract from each image.
}

# Convert the theta angles from degrees to radians for GLCM and GLRLM.
extractionParams["GLCM"]["theta"] = np.radians(extractionParams["GLCM"]["theta"])
extractionParams["GLRLM"]["theta"] = np.radians(extractionParams["GLRLM"]["theta"])

# Define the dataset path.
# Ensure to change this path to the actual location of your downloaded dataset.
datasetPath = r"../../Datasets/COVID-19 Radiography Database"
# Extract the feature names that are turned on from the extractionParams dictionary.
turnedOnFeatures = [
  key for key, value in extractionParams.items()
  if (isinstance(value, dict) and value.get("turnOn", False))
]
# Convert the feature names to a string format suitable for the file name.
turnedOnFeaturesStr = "-".join(turnedOnFeatures)
# Define the storage path for the features.
featuresPath = rf"Data/COVID-19 Radiography Database ({turnedOnFeaturesStr}) Features.csv"

# List all classes in the dataset directory to process.
classes = [
  cls for cls in os.listdir(datasetPath)
  if os.path.isdir(os.path.join(datasetPath, cls))
]
# If you are not sure about the classes, you can define them manually:
# classes = ["COVID-19", "Normal", "Lung_Opacity", "Viral Pneumonia"]

# Create a list to store the features extracted from each image.
history = []

# Iterate through each class to process images.
for cls in tqdm.tqdm(classes, desc="Processing classes."):
  # Get the path for the current class.
  clsPath = os.path.join(datasetPath, cls)

  # If you remember, in the first project, we had masks for each image where
  # the mask had "_mask" in its name as a postfix. Also, both the images
  # and masks were stored in the same directory.
  # List all files in the class directory.
  # files = os.listdir(clsPath)
  # Filter out files that are images (not masks).
  # files = [f for f in files if "_mask" not in f]

  # In this project, the situation is different, as the images are stored in the
  # "images" subdirectory and the masks are stored in the "masks" subdirectory.
  # Also, both the images and masks have the same name and extension.
  # So, you will need to adjust the file paths and the preprocessing steps accordingly
  # based on the dataset in hand.
  # List all files in the class and "images" subdirectory.
  files = os.listdir(os.path.join(clsPath, "images"))

  # Iterate through each file in the class directory.
  for file in tqdm.tqdm(files, desc=f"Processing class {cls}."):
    # Get the full path of the file.
    filePath = os.path.join(clsPath, "images", file)

    # Find the corresponding mask file.
    maskPath = os.path.join(clsPath, "masks", file)

    # Read the image and mask.
    caseImg = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    caseSeg = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

    # Check if the image and mask are read correctly.
    if ((caseImg is None) or (caseSeg is None)):
      print(f"Error reading image or mask for file: {filePath} or {maskPath}. Skipping.")
      continue

    regions = ExtractMultipleObjectsFromROI(
      caseImg, caseSeg,
      targetSize=extractionParams["targetSize"],  # Resize the image to the target size.
      cntAreaThreshold=0,  # Skip contours smaller than threshold to ignore noise/artifacts.
      sortByX=True,  # Sort the regions by their x-coordinate to ensure consistent ordering during feature extraction.
    )

    if ((len(regions) == 0) or (len(regions) > extractionParams["maxRegions"])):
      # If no regions are found or more than the maximum allowed regions, skip this image.
      continue

    # We have multiple regions in the image, so we will iterate through each region.
    # For each region, we will extract the different features that got enabled through the
    # extractionParams dictionary and "turnOn" flags.
    # If there are multiple elements in the lists, we will iterate through each element.
    # For example, if the GLCM distance is [1, 2], we will calculate the GLCM features
    # for both distances and store them in the same dictionary.
    # To distinguish between the features, we will add the distance or angle to the feature name.

    imageFeatures = {}  # Dictionary to store features for the current image.
    for i, region in enumerate(regions):
      regionFeatures = {}  # Dictionary to store features for the current region.
      if (extractionParams["FirstOrderFeatures"]["turnOn"]):
        # If first order features are enabled, we will calculate them.
        # Calculate first order features.
        # FirstOrderFeatures2DV2 is a new version of the function that
        # accepts the region/data directly instead of the full image and mask.
        firstOrderFeatures, hist2D = FirstOrderFeatures2DV2(
          region,  # The cropped region of interest (ROI) image.
          isNorm=extractionParams["isNorm"],  # Whether to normalize the features.
          ignoreZeros=extractionParams["ignoreZeros"],  # Whether to ignore zero-valued pixels.
        )
        # Store the first order features in the regionFeatures dictionary.
        regionFeatures.update(firstOrderFeatures)  # Update the dictionary with first order features.

      if (extractionParams["GLCM"]["turnOn"]):
        # If GLCM features are enabled, we will calculate them.
        for d in extractionParams["GLCM"]["d"]:
          for theta in extractionParams["GLCM"]["theta"]:
            # Calculate the GLCM co-occurrence matrix.
            coMatrix = CalculateGLCMCooccuranceMatrix(
              region,  # The cropped ROI image.
              d,  # Distance between voxel pairs.
              theta,  # Angle (in radians) for the direction of voxel pairs.
              isSymmetric=extractionParams["GLCM"]["isSymmetric"],  # Whether to make the GLCM symmetric.
              isNorm=extractionParams["isNorm"],  # Whether to normalize the features.
              ignoreZeros=extractionParams["ignoreZeros"],  # Whether to ignore zero-valued pixels.
            )
            # Calculate GLCM features from the co-occurrence matrix.
            glcmFeatures = CalculateGLCMFeaturesOptimized(coMatrix)
            # Update the regionFeatures dictionary with GLCM features.
            regionFeatures.update({
              f"{key}_D{d}_T{np.degrees(theta)}": value
              for key, value in glcmFeatures.items()
            })

      if (extractionParams["GLRLM"]["turnOn"]):
        # If GLRLM features are enabled, we will calculate them.
        for theta in extractionParams["GLRLM"]["theta"]:
          # Calculate the GLRLM matrix.
          rlMatrix = CalculateGLRLMRunLengthMatrix(
            region,  # The cropped ROI image.
            theta,  # Angle (in radians) for the direction of voxel pairs.
            isNorm=extractionParams["isNorm"],  # Whether to normalize the features.
            ignoreZeros=extractionParams["ignoreZeros"],  # Whether to ignore zero-valued pixels.
          )
          # Calculate the GLRLM features.
          glrlmFeatures = CalculateGLRLMFeatures(
            rlMatrix,  # The run-length matrix.
            region,  # The cropped ROI image.
          )
          # Update the regionFeatures dictionary with GLRLM features.
          regionFeatures.update({
            f"{key}_T{np.degrees(theta)}": value
            for key, value in glrlmFeatures.items()
          })

      if (extractionParams["GLSZM"]["turnOn"]):
        # If GLSZM features are enabled, we will calculate them.
        for connectivity in extractionParams["GLSZM"]["connectivity"]:
          # Calculate the GLSZM matrix.
          szMatrix, szDict, N, Z = CalculateGLSZMSizeZoneMatrix(
            region,  # The cropped ROI image.
            connectivity=connectivity,  # Connectivity type (4 or 8).
            isNorm=extractionParams["isNorm"],  # Whether to normalize the features.
            ignoreZeros=extractionParams["ignoreZeros"],  # Whether to ignore zero-valued pixels.
          )
          # Calculate the GLSZM features.
          glszmFeatures = CalculateGLSZMFeatures(
            szMatrix,  # The size zone matrix.
            region,  # The cropped ROI image.
            N,
            Z,
          )
          # Update the regionFeatures dictionary with GLSZM features.
          regionFeatures.update({
            f"{key}_C{connectivity}": value
            for key, value in glszmFeatures.items()
          })

      # After calculating all features for the current region, we will update the imageFeatures dictionary.
      if (len(regionFeatures) == 0):
        # If no features were extracted, skip this region.
        continue

      # Update the imageFeatures dictionary with the region features.
      imageFeatures.update({
        f"R{i + 1}_{key}": value  # Prefix the feature name with the region index.
        for key, value in regionFeatures.items()
      })

    # Append the features to the history list.
    history.append({
      "File" : file,  # Store the file name.
      **imageFeatures,  # Unpack the image features into the dictionary.
      "Class": cls,  # Store the class name.
    })

# Convert the history list to a DataFrame.
featuresDF = pd.DataFrame(history)
# Save the features DataFrame to a CSV file.
featuresDF.to_csv(
  featuresPath,  # Path to save the CSV file.
  index=False,  # Do not include the index in the CSV file.
)

print("Feature extraction completed successfully.")
