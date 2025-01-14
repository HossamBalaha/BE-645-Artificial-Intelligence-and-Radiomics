# Author: Hossam Magdy Balaha
# Date: June 19th, 2024
# Permissions and Citation: Refer to the README file.

import cv2
import numpy as np

caseImgPath = r"Sample Liver Image.bmp"
caseSegPath = r"Sample Liver Segmentation.bmp"

# Load the images.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Extract the ROI.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Crop the ROI.
x, y, w, h = cv2.boundingRect(roi)
cropped = roi[y:y + h, x:x + w]

# Calculate the Shape Features:

# 1. Area.
area = cv2.countNonZero(cropped)

# 2. Perimeter.
contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largestContour = max(contours, key=cv2.contourArea)
perimeter = cv2.arcLength(largestContour, True)

# 3. Centroid.
moments = cv2.moments(largestContour)
centroidX = int(moments["m10"] / moments["m00"])
centroidY = int(moments["m01"] / moments["m00"])

# 4. Bounding Box.
x, y, w, h = cv2.boundingRect(largestContour)

# 5. Aspect Ratio.
aspectRatio = w / h

# 6. Compactness.
compactness = (perimeter ** 2) / (4 * np.pi * area)

# 7. Eccentricity.
mu20 = moments["mu20"] / moments["m00"]
mu02 = moments["mu02"] / moments["m00"]
eccentricity = np.sqrt(1 - (mu02 / mu20))

# 8. Convex Hull.
smallestConvexHull = cv2.convexHull(largestContour)
convexHullArea = cv2.contourArea(smallestConvexHull)

# 9. Extent.
extent = area / (w * h)

# 10. Solidity.
solidity = area / convexHullArea

# 11. Major Axis Length.
majorAxisLength = 2 * np.sqrt(moments["m20"] / moments["m00"])

# 12. Minor Axis Length.
minorAxisLength = 2 * np.sqrt(moments["m02"] / moments["m00"])

# 13. Orientation.
orientation = 0.5 * np.arctan2(2 * moments["mu11"], moments["mu20"] - moments["mu02"])

# 14. Roundness.
roundness = (4 * area) / (np.pi * perimeter ** 2)

print("Area: ", area)
print("Perimeter: ", perimeter)
print("Centroid: ", centroidX, centroidY)
print("Bounding Box: ", x, y, w, h)
print("Aspect Ratio: ", aspectRatio)
print("Compactness: ", compactness)
print("Eccentricity: ", eccentricity)
print("Convex Hull Area: ", convexHullArea)
print("Extent: ", extent)
print("Solidity: ", solidity)
print("Major Axis Length: ", majorAxisLength)
print("Minor Axis Length: ", minorAxisLength)
print("Orientation: ", orientation)
print("Roundness: ", roundness)
