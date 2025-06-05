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
import numpy as np  # For numerical operations.
import matplotlib.pyplot as plt  # For plotting graphs.

# Define a 2D array (matrix) of pixel values.
X = np.array([
  [1, 2, 4, 2, 1],
  [3, 1, 2, 4, 5],
  [5, 3, 2, 2, 4],
  [1, 4, 1, 2, 4],
  [3, 2, 2, 1, 5],
]).astype(np.uint8)  # Convert the array to unsigned 8-bit integers.

# Another example 2D array (matrix) of pixel values.
# X = np.array([
#   [5, 2, 4, 2, 5],
#   [3, 1, 2, 4, 5],
#   [5, 3, 5, 2, 4],
#   [1, 4, 1, 2, 4],
#   [5, 5, 5, 1, 5],
# ]).astype(np.uint8)  # Convert the array to unsigned 8-bit integers.

# Another example 2D array (matrix) of pixel values.
# X = np.array([
#   [1, 2, 5, 6, 4],
#   [3, 2, 5, 6, 1],
#   [4, 5, 3, 2, 1],
#   [6, 4, 1, 2, 3],
#   [1, 4, 5, 2, 3],
# ]).astype(np.uint8)  # Convert the array to unsigned 8-bit integers.

# Calculate the histogram of the 2D array.
min = int(np.min(X))  # Find the minimum value in the array.
max = int(np.max(X))  # Find the maximum value in the array.
hist2D = []  # Initialize an empty list to store the histogram values.

# Loop through each possible value in the range [min, max].
for i in range(min, max + 1):
  hist2D.append(np.count_nonzero(X == i))  # Count occurrences of the value `i` in the array.
hist2D = np.array(hist2D)  # Convert the histogram list to a NumPy array.

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
print("Min:", min)  # Print the minimum value.
print("Max:", max)  # Print the maximum value.
print("Range:", rng)  # Print the range of values.
print("Count:", count)  # Print the total count of values.
print("Sum:", sum)  # Print the sum of values.
print("Mean:", np.round(mean, 4))  # Print the mean value.
print("Variance:", np.round(variance, 4))  # Print the variance.
print("Standard Deviation:", np.round(stdDev, 4))  # Print the standard deviation.
print("Skewness:", np.round(skewness, 4))  # Print the skewness.
print("Kurtosis:", np.round(kurtosis, 4))  # Print the kurtosis.
print("Excess Kurtosis:", np.round(exKurtosis, 4))  # Print the excess kurtosis.

# Plot the histogram.
plt.figure()  # Create a new figure for the plot.
plt.bar(rng, hist2D)  # Plot the histogram as a bar chart.
plt.title("2D Histogram")  # Set the title of the plot.
plt.xlabel("Pixel Value")  # Label the x-axis.
plt.ylabel("Frequency")  # Label the y-axis.
plt.tight_layout()  # Adjust the layout for better visualization.
plt.savefig(
  "Data/2D_Histogram.jpg",  # Save the plot as an image file.
  dpi=300,  # Set the resolution of the plot.
  bbox_inches="tight",  # Save the plot as an image file.
)  # Save the plot as an image file.

plt.show()  # Display the histogram plot.
plt.close()  # Close the figure.
plt.clf()  # Clear the current figure.
