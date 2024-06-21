# Author: Hossam Magdy Balaha
# Date: May 20th, 2024
# Permissions and Citations: Refer to the README file.

import cv2
import matplotlib.pyplot as plt

caseImgPath = r"Sample Liver Image.bmp"
caseSegPath = r"Sample Liver Segmentation.bmp"

# Load the images.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)

# Get the shape of the images.
caseImgShape = caseImg.shape
caseSegShape = caseSeg.shape

print("Image Shape: ", caseImgShape)
print("Segmentation Shape: ", caseSegShape)

# Extract the ROI.
roi = cv2.bitwise_and(caseImg, caseSeg)

# Crop the ROI.
x, y, w, h = cv2.boundingRect(roi)
cropped = roi[y:y + h, x:x + w]

# Display the images.
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(caseImg, cmap="gray")
plt.title("Image")
plt.axis("off")
plt.tight_layout()
plt.subplot(2, 2, 2)
plt.imshow(caseSeg, cmap="gray")
plt.title("Segmentation")
plt.axis("off")
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.imshow(roi, cmap="gray")
plt.title("ROI")
plt.axis("off")
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.imshow(cropped, cmap="gray")
plt.title("Cropped ROI")
plt.axis("off")
plt.tight_layout()
plt.show()
