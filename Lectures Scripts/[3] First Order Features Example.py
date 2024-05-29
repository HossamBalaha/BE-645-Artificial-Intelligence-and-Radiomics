# Author: Hossam Magdy Balaha
# Date: May 20th, 2024

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
  [1, 2, 4, 2, 1],
  [3, 1, 2, 4, 5],
  [5, 3, 2, 2, 4],
  [1, 4, 1, 2, 4],
  [3, 2, 2, 1, 5],
]).astype(np.uint8)

# Calculate the histogram.
min = int(np.min(X))
max = int(np.max(X))
hist2D = []
for i in range(min, max + 1):
  hist2D.append(np.count_nonzero(X == i))
hist2D = np.array(hist2D)

# Calculate the count from the histogram.
count = np.sum(hist2D)

# Determine the range.
rng = np.arange(min, max + 1)

# Calculate the sum from the histogram.
sum = np.sum(hist2D * rng)

# Calculate the mean from the histogram.
mean = sum / count

# Calculate the variance from the histogram.
variance = np.sum(hist2D * (rng - mean) ** 2) / count

# Calculate the standard deviation from the histogram.
stdDev = np.sqrt(variance)

# Calculate the skewness from the histogram.
skewness = np.sum(hist2D * (rng - mean) ** 3) / (count * stdDev ** 3)

# Calculate the kurtosis from the histogram.
kurtosis = np.sum(hist2D * (rng - mean) ** 4) / (count * stdDev ** 4)

# Calculate the excess kurtosis from the histogram.
exKurtosis = kurtosis - 3

# Print the results.
print("Min: ", min)
print("Max: ", max)
print("Range: ", rng)
print("Count: ", count)
print("Sum: ", sum)
print("Mean: ", mean)
print("Variance: ", variance)
print("Standard Deviation: ", stdDev)
print("Skewness: ", skewness)
print("Kurtosis: ", kurtosis)
print("Excess Kurtosis: ", exKurtosis)

# Plot the histogram.
plt.figure()
plt.bar(np.arange(min, max + 1), hist2D)
plt.title("2D Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
