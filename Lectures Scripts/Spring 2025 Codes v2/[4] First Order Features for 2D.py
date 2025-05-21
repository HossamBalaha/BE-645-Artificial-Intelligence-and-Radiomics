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
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.

# Define the paths to the input image and segmentation mask.
caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the liver image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the liver segmentation mask.

# Load the images in grayscale mode.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the liver image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Extract the Region of Interest (ROI) using the segmentation mask.
roi = cv2.bitwise_and(caseImg, caseSeg)  # Apply bitwise AND operation to extract the ROI.

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

# Calculate the total count of values from the histogram.
count = np.sum(hist2D)  # Sum all frequencies in the histogram.

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

# Print the calculated statistics.
print("Min: ", min)  # Print the minimum value.
print("Max: ", max)  # Print the maximum value.
print("Range: ", rng)  # Print the range of values.
print("Count: ", count)  # Print the total count of values.
print("Sum: ", sum)  # Print the sum of values.
print("Mean: ", np.round(mean, 4))  # Print the mean value.
print("Variance: ", np.round(variance, 4))  # Print the variance.
print("Standard Deviation: ", np.round(stdDev, 4))  # Print the standard deviation.
print("Skewness: ", np.round(skewness, 4))  # Print the skewness.
print("Kurtosis: ", np.round(kurtosis, 4))  # Print the kurtosis.
print("Excess Kurtosis: ", np.round(exKurtosis, 4))  # Print the excess kurtosis.

# Plot the histogram.
plt.figure()  # Create a new figure for the plot.
plt.bar(np.arange(min, max + 1), hist2D)  # Plot the histogram as a bar chart.
plt.title("2D Histogram")  # Set the title of the plot.
plt.xlabel("Pixel Value")  # Label the x-axis.
plt.ylabel("Frequency")  # Label the y-axis.
plt.tight_layout()  # Adjust the layout for better visualization.

# Save the histogram plot as an image file.
ext = caseImgPath.split(".")[-1]  # Get the file extension of the input image.
plt.savefig(
  caseImgPath.replace(f".{ext}", f" Histogram.{ext}"),  # Replace the file extension and save the plot.
  dpi=300,  # Set the resolution of the saved image.
  bbox_inches="tight",  # Ensure the entire plot is saved without cropping.
)

plt.show()  # Display the histogram plot.
plt.close()  # Close the plot to free up memory.
