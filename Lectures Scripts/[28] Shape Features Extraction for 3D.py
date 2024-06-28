# Author: Hossam Magdy Balaha
# Date: June 28th, 2024
# Permissions and Citation: Refer to the README file.

import cv2, trimesh
import numpy as np


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


contentRng = (45, 73)

caseImgPaths = [
  rf"Volume Slices/Volume Slice {i}.bmp"
  for i in range(contentRng[0], contentRng[1] + 1)
]
caseSegPaths = [
  rf"Segmentation Slices/Segmentation Slice {i}.bmp"
  for i in range(contentRng[0], contentRng[1] + 1)
]

volumeCropped = ReadVolume(caseImgPaths, caseSegPaths)

features = ShapeFeatures3D(volumeCropped)

# Print the shape features.
for key in features:
  print(key, ":", features[key])

mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volumeCropped)
scene = mesh.scene()
scene.show(resolution=(500, 500))
