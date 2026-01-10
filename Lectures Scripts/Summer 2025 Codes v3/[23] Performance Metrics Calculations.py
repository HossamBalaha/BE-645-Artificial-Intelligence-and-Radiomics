'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 9th, 2024
# Last Modification Date: Jun 20th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import numpy as np  # For numerical operations.
from HMB_Summer_2025_Helpers import *  # Import custom helper functions.

# Example 1: Balanced classes.
confMatrix = [
  [50, 2, 1, 2],
  [3, 45, 5, 2],
  [1, 2, 40, 7],
  [0, 1, 3, 51],
]

# Example 2: Imbalanced classes.
# confMatrix = [
#   [90, 3, 1, 1],
#   [4, 120, 8, 3],
#   [2, 7, 50, 12],
#   [3, 5, 10, 25],
# ]

# Example 3: Binary classification.
# confMatrix = [
#   [90, 10],
#   [5, 95],
# ]

# Example 4: Binary classification with imbalanced classes.
# confMatrix = [
#   [90, 10],
#   [15, 85],
# ]

# Example 5: Multiclass classification.
# confMatrix = [
#   [90, 3, 1, 1],
#   [4, 120, 8, 3],
#   [2, 7, 50, 12],
#   [3, 5, 10, 25],
# ]

# Example 6: Multiclass classification with imbalanced classes.
# confMatrix = [
#   [2424, 14, 234, 33, 45],
#   [93, 1986, 276, 34, 92],
#   [257, 141, 2549, 53, 250],
#   [101, 30, 66, 1966, 87],
#   [212, 63, 87, 73, 3315],
# ]

# Example 7: Multiclass classification with imbalanced classes.
# confMatrix = [
#   [25, 1, 2, 0],
#   [3, 35, 3, 3],
#   [2, 3, 25, 5],
#   [2, 3, 3, 30],
# ]


# Calculate the performance metrics from the confusion matrix.
performanceMetrics = CalculatePerformanceMetrics(confMatrix)

# Print the calculated performance metrics.
for metric, value in performanceMetrics.items():
  print(f"{metric}: {np.round(value, 4)}")
