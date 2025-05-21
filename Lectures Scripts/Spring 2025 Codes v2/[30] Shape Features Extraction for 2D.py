'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 19th, 2024
# Last Modification Date: Feb 25th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os
import cv2
import numpy as np


def ShapeFeatures(matrix):
  """
  Calculate shape features of a given binary matrix.
  The function computes various shape features such as area, perimeter,
  centroid, bounding box, aspect ratio, compactness, eccentricity,
  convex hull area, extent, solidity, major and minor axis lengths,
  orientation, and roundness.
  Args:
    matrix: A matrix representing the binary image or segmented region.
  Returns:
    A dictionary containing the calculated shape features.
  """
  # Check if the input matrix is empty or not.
  if (matrix is None or matrix.size == 0):
    # Raise error if the matrix is empty.
    raise ValueError("The input matrix is empty. Please provide a valid matrix.")

  # Calculate the Shape Features:

  # 1. Area.
  # Counts the number of non-zero pixels in the cropped image.
  area = cv2.countNonZero(matrix)

  # 2. Perimeter.
  # Finds contours in the matrix image.
  contours, _ = cv2.findContours(matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Identifies the largest contour.
  largestContour = max(contours, key=cv2.contourArea)
  # Calculates the perimeter of the largest contour.
  perimeter = cv2.arcLength(largestContour, True)

  # 3. Centroid.
  # Computes moments of the matrix.
  moments = cv2.moments(matrix)
  # Calculates the X-coordinate of the centroid.
  centroidX = int(moments["m10"] / moments["m00"])
  # Calculates the Y-coordinate of the centroid.
  centroidY = int(moments["m01"] / moments["m00"])

  # 4. Bounding Box.
  # Recalculates the bounding box for the matrix.
  x, y, w, h = cv2.boundingRect(matrix)

  # 5. Aspect Ratio.
  # Computes the aspect ratio of the bounding box.
  aspectRatio = w / h

  # 6. Compactness.
  # Calculates compactness using perimeter and area.
  compactness = (perimeter ** 2) / (4 * np.pi * area)

  # 7. Eccentricity.
  # Computes normalized second-order moment mu20.
  mu20 = moments["mu20"] / moments["m00"]
  # Computes normalized second-order moment mu02.
  mu02 = moments["mu02"] / moments["m00"]
  # Calculates eccentricity based on moments.
  eccentricity = np.sqrt(1 - (mu02 / mu20))

  # 8. Convex Hull.
  # Finds the convex hull of the largest contour.
  smallestConvexHull = cv2.convexHull(largestContour)
  # Calculates the area of the convex hull.
  convexHullArea = cv2.contourArea(smallestConvexHull)

  # 9. Extent (or Rectangularity).
  # Computes the extent as the ratio of contour area to bounding box area.
  extent = area / (w * h)

  # 10. Solidity.
  # Calculates solidity as the ratio of contour area to convex hull area.
  solidity = area / convexHullArea

  # 11. Major Axis Length.
  # Computes the length of the major axis using the second-order moment mu20.
  majorAxisLength = 2 * np.sqrt(moments["m20"] / moments["m00"])

  # 12. Minor Axis Length.
  # Computes the length of the minor axis using the second-order moment mu02.
  minorAxisLength = 2 * np.sqrt(moments["m02"] / moments["m00"])

  # 13. Orientation.
  # Calculates orientation angle as the angle of the major axis of the ellipse.
  orientation = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

  # 14. Roundness.
  # Computes roundness based on area and perimeter.
  roundness = (4 * area) / (np.pi * perimeter ** 2)

  # 15. Symmetry.
  # Flip the matrix horizontally and vertically.
  flippedHorizontal = np.fliplr(matrix)
  flippedVertical = np.flipud(matrix)
  # Calculate the symmetry score for horizontal flipping.
  horizontalSymmetry = np.sum(matrix == flippedHorizontal) / area
  # Calculate the symmetry score for vertical flipping.
  verticalSymmetry = np.sum(matrix == flippedVertical) / area
  # Calculate the average symmetry score.
  symmetry = (horizontalSymmetry + verticalSymmetry) / 2.0

  # 16. Elongation.
  # Calculate the elongation based on the major and minor axis lengths.
  elongation = majorAxisLength / minorAxisLength

  # 17. Thinness Ratio.
  # Calculate the thinness ratio based on the perimeter and area.
  thinnessRatio = np.power(perimeter, 2) / area

  # 18. Convexity.
  # Convexity measures how close the shape is to being convex.
  # It is the ratio of the perimeter of the convex hull to the perimeter of the shape.
  # True: The contour is closed, False: The contour is open.
  convexHullPerimeter = cv2.arcLength(smallestConvexHull, True)
  convexity = convexHullPerimeter / perimeter

  # 19. Sparseness.
  # Sparseness measures how "spread out" the shape is.
  # Calculate the area of the bounding box.
  boundingBoxArea = w * h
  # Compute sparseness as a measure of spread.
  sparseness = (np.sqrt(area / boundingBoxArea) - (area / boundingBoxArea))

  # 20. Curvature.
  # Curvature measures how sharply the contour bends at each point.
  curvatures = []
  for i in range(len(largestContour)):
    # Loop through all points in the largest contour.
    p1 = largestContour[i - 1][0]  # Previous point.
    p2 = largestContour[i][0]  # Current point.
    p3 = largestContour[(i + 1) % len(largestContour)][0]  # Next point.
    # Calculate the curvature using the cross product and dot product.
    v1 = p2 - p1  # Vector from p1 to p2.
    v2 = p3 - p2  # Vector from p2 to p3.
    crossProduct = np.cross(v1, v2)  # Cross product of the vectors.
    dotProduct = np.dot(v1, v2)  # Dot product of the vectors.
    angle = np.arctan2(crossProduct, dotProduct)  # Angle between the vectors.
    curvatures.append(angle)  # Append the curvature to the list.
  # Calculate the average curvature.
  averageCurvature = np.mean(curvatures)
  # Calculate the standard deviation of curvature.
  stdCurvature = np.std(curvatures)

  # Return all calculated features as a dictionary.
  return {
    "Area"               : area,
    "Perimeter"          : perimeter,
    "Centroid X"         : centroidX,
    "Centroid Y"         : centroidY,
    "Bounding Box X"     : x,
    "Bounding Box Y"     : y,
    "Bounding Box W"     : w,
    "Bounding Box H"     : h,
    "Aspect Ratio"       : aspectRatio,
    "Compactness"        : compactness,
    "Eccentricity"       : eccentricity,
    "Convex Hull Area"   : convexHullArea,
    "Extent"             : extent,
    "Solidity"           : solidity,
    "Major Axis Length"  : majorAxisLength,
    "Minor Axis Length"  : minorAxisLength,
    "Orientation"        : orientation,
    "Roundness"          : roundness,
    "Horizontal Symmetry": horizontalSymmetry,
    "Vertical Symmetry"  : verticalSymmetry,
    "Symmetry"           : symmetry,
    "Elongation"         : elongation,
    "Thinness Ratio"     : thinnessRatio,
    "Convexity"          : convexity,
    "Sparseness"         : sparseness,
    "Curvature"          : averageCurvature,
    "Std Curvature"      : stdCurvature,
  }


# Define path to sample liver image file.
caseImgPath = r"Data/Sample Liver Image.bmp"
# Define path to corresponding liver segmentation mask file.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"

# Verify both image and mask files exist before attempting to load.
if (not os.path.exists(caseImgPath)) or (not os.path.exists(caseSegPath)):
  # Raise error with descriptive message if any file is missing.
  raise FileNotFoundError("One or more files were not found. Please check the file paths.")

# Load input image as grayscale (single channel intensity values).
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
# Load segmentation mask as grayscale (binary or labeled format).
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Extract Region of Interest (ROI) by masking input image with segmentation.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Calculate bounding box coordinates of non-zero region in ROI.
x, y, w, h = cv2.boundingRect(roi)
# Crop ROI to tight bounding box around segmented area.
cropped = roi[y:y + h, x:x + w]

# Validate cropped image contains non-zero pixels to prevent empty processing.
if (np.sum(cropped) <= 0):
  # Raise error if cropped image is completely black/empty.
  raise ValueError("The cropped image is empty. Please check the segmentation mask.")

# Call the ShapeFeatures function to calculate shape features.
shapeFeatures = ShapeFeatures(cropped)

# Print the calculated shape features.
print("Shape Features:")
for feature, value in shapeFeatures.items():
  print(f"{feature}: {value:0.4f}")
