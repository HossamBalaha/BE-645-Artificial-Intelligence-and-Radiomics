# Author: Hossam Magdy Balaha
# Date: June 9th, 2024
# Adopted from: https://github.com/HossamBalaha/BE-544-Artificial-Intelligence-Techniques-in-Digital-Histopathology
# Permissions and Citation: Refer to the README file.

import numpy as np

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

confMatrix = np.array(confMatrix)

# Calculate TP, TN, FP, FN.
TP = np.diag(confMatrix)
FP = np.sum(confMatrix, axis=0) - TP
FN = np.sum(confMatrix, axis=1) - TP
TN = np.sum(confMatrix) - (TP + FP + FN)

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

# Calculate all metrics.

# Using macro averaging.
precision = np.mean(TP / (TP + FP))
recall = np.mean(TP / (TP + FN))
f1 = 2 * precision * recall / (precision + recall)
accuracy = np.sum(TP) / np.sum(confMatrix)
specificity = np.mean(TN / (TN + FP))

print("Macro Precision:", precision)
print("Macro Recall:", recall)
print("Macro F1:", f1)
print("Macro Accuracy:", accuracy)
print("Macro Specificity:", specificity)

# Using micro averaging.
precision = np.sum(TP) / np.sum(TP + FP)
recall = np.sum(TP) / np.sum(TP + FN)
f1 = 2 * precision * recall / (precision + recall)
accuracy = np.sum(TP) / np.sum(confMatrix)
specificity = np.sum(TN) / np.sum(TN + FP)

print("Micro Precision:", precision)
print("Micro Recall:", recall)
print("Micro F1:", f1)
print("Micro Accuracy:", accuracy)
print("Micro Specificity:", specificity)

# Using weighted averaging.
samples = np.sum(confMatrix, axis=1)
weights = samples / np.sum(confMatrix)
print("Samples:", samples)
print("Weights:", weights)

precision = np.sum(TP / (TP + FP) * weights)
recall = np.sum(TP / (TP + FN) * weights)
f1 = 2 * precision * recall / (precision + recall)
accuracy = np.sum(TP) / np.sum(confMatrix)
specificity = np.sum(TN / (TN + FP) * weights)

print("Weighted Precision:", precision)
print("Weighted Recall:", recall)
print("Weighted F1:", f1)
print("Weighted Accuracy:", accuracy)
print("Weighted Specificity:", specificity)
