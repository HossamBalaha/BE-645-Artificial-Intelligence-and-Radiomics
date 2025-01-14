# Author: Hossam Magdy Balaha
# Date: May 20th, 2024
# Permissions and Citations: Refer to the README file.

import cv2, os
import numpy as np
import nibabel as nib

caseVolPath = r"Sample Volume.nii"
caseSegPath = r"Sample Segmentation.nii"

# Load the NII files.
caseVol = nib.load(caseVolPath)
caseSeg = nib.load(caseSegPath)

# Get the data from the NII files.
caseVolData = caseVol.get_fdata()
caseSegData = caseSeg.get_fdata()

# Get the shape of the data.
caseVolShape = caseVolData.shape
caseSegShape = caseSegData.shape

print("Volume Shape: ", caseVolShape)
print("Segmentation Shape: ", caseSegShape)

storageVolFolder = r"Volume Slices"
storageSegFolder = r"Segmentation Slices"

# Create the folders if they don't exist.
if not os.path.exists(storageVolFolder):
  os.makedirs(storageVolFolder)
if not os.path.exists(storageSegFolder):
  os.makedirs(storageSegFolder)

# Extract the slices from the volume and the segmentation.
for i in range(caseVolShape[2]):
  volSlice = caseVolData[:, :, i]
  segSlice = caseSegData[:, :, i]

  # Rotate the slices.
  volSlice = np.rot90(volSlice)
  segSlice = np.rot90(segSlice)

  segSlice[segSlice == 1] = 127
  segSlice[segSlice == 2] = 255

  # Save the slices.
  cv2.imwrite(
    os.path.join(storageVolFolder, f"Volume Slice {i}.bmp"),
    volSlice
  )
  cv2.imwrite(
    os.path.join(storageSegFolder, f"Segmentation Slice {i}.bmp"),
    segSlice
  )
