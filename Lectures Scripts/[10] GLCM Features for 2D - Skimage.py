# Author: Hossam Magdy Balaha
# Date: May 29th, 2024
# Permission:
# - Work can be shared but not used for commercial purposes.
# - Author name and citation must be included.
# - Modifications must be documented.
# Citation:
# Balaha, H. M. (2024). BE 645 Artificial Intelligence (AI) and Radiomics (Summer 2024) (Version 1.06.19) [Computer software]. https://github.com/HossamBalaha/BE-645-Artificial-Intelligence-and-Radiomics

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

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

d = 1
theta = 0
N = np.max(cropped) + 1

# Calculate the co-matrix.
coMatrix = graycomatrix(
  cropped,
  distances=[d],
  angles=[theta],
  levels=N,
  symmetric=False,
  normed=True,
)

# Extract GLCM features.
contrast = graycoprops(coMatrix[1:, 1:, :, :], "contrast")[0, 0]
correlation = graycoprops(coMatrix[1:, 1:, :, :], "correlation")[0, 0]
energy = graycoprops(coMatrix[1:, 1:, :, :], "energy")[0, 0]
homogeneity = graycoprops(coMatrix[1:, 1:, :, :], "homogeneity")[0, 0]
ASM = graycoprops(coMatrix[1:, 1:, :, :], "ASM")[0, 0]

# Print the GLCM features.
print("Contrast: ", contrast)
print("Correlation: ", correlation)
print("Energy: ", energy)
print("Homogeneity: ", homogeneity)
print("ASM: ", ASM)

# Display the co-occurrence matrix.
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cropped, cmap="gray")
plt.title("Cropped Image")
plt.axis("off")
plt.colorbar()
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.imshow(coMatrix[1:, 1:, 0, 0], cmap="gray")
plt.title("Co-occurrence Matrix")
plt.colorbar()
plt.tight_layout()
plt.show()
