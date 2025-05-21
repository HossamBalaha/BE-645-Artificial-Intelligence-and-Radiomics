'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 9th, 2024
# Last Modification Date: Feb 4th, 2025
# Permissions and Citation: Refer to the README file.
'''

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

# Convert the confusion matrix to a NumPy array for easier manipulation.
confMatrix = np.array(confMatrix)

# Check if the confusion matrix is for binary classification.
noOfClasses = confMatrix.shape[0]
if (noOfClasses > 2):
  # Calculate True Positives (TP) as the diagonal elements of the confusion matrix.
  TP = np.diag(confMatrix)
  # Calculate False Positives (FP) as the sum of each column minus the TP.
  FP = np.sum(confMatrix, axis=0) - TP
  # Calculate False Negatives (FN) as the sum of each row minus the TP.
  FN = np.sum(confMatrix, axis=1) - TP
  # Calculate True Negatives (TN) as the total sum of the matrix minus TP, FP, and FN.
  TN = np.sum(confMatrix) - (TP + FP + FN)
else:
  # For binary classification, the confusion matrix is a 2x2 matrix.
  # Unravel the confusion matrix to get the TP, FP, FN, and TN.
  # The order of the elements is TN, FP, FN, TP.
  TN, FP, FN, TP = confMatrix.ravel()

# Avoid division by zero by adding a small epsilon value.
eps = 1e-10
TP = TP + eps
FP = FP + eps
FN = FN + eps
TN = TN + eps

# Print the calculated TP, FP, FN, and TN.
print("TP:", np.round(TP, 4))
print("FP:", np.round(FP, 4))
print("FN:", np.round(FN, 4))
print("TN:", np.round(TN, 4))

# Calculate precision using macro averaging: mean of TP / (TP + FP) for each class.
precision = np.mean(TP / (TP + FP))
# Calculate recall using macro averaging: mean of TP / (TP + FN) for each class.
recall = np.mean(TP / (TP + FN))
# Calculate F1 score using macro averaging: harmonic mean of precision and recall.
f1 = 2 * precision * recall / (precision + recall)
# Calculate accuracy using macro averaging: sum of TP and TN divided by the total sum of the matrix.
accuracy = np.mean(TP + TN) / np.sum(confMatrix)
# Calculate specificity using macro averaging: mean of TN / (TN + FP) for each class.
specificity = np.mean(TN / (TN + FP))

# Print the macro-averaged metrics.
print("Macro Precision:", np.round(precision, 4))
print("Macro Recall:", np.round(recall, 4))
print("Macro F1:", np.round(f1, 4))
print("Macro Accuracy:", np.round(accuracy, 4))
print("Macro Specificity:", np.round(specificity, 4))

# Calculate precision using micro averaging: sum of TP divided by the sum of TP and FP.
precision = np.sum(TP) / np.sum(TP + FP)
# Calculate recall using micro averaging: sum of TP divided by the sum of TP and FN.
recall = np.sum(TP) / np.sum(TP + FN)
# Calculate F1 score using micro averaging: harmonic mean of precision and recall.
f1 = 2 * precision * recall / (precision + recall)
# Calculate accuracy using micro averaging: sum of TP and TN divided by TP, TN, FP, and FN.
accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
# Calculate specificity using micro averaging: sum of TN divided by the sum of TN and FP.
specificity = np.sum(TN) / np.sum(TN + FP)

# Print the micro-averaged metrics.
print("Micro Precision:", np.round(precision, 4))
print("Micro Recall:", np.round(recall, 4))
print("Micro F1:", np.round(f1, 4))
print("Micro Accuracy:", np.round(accuracy, 4))
print("Micro Specificity:", np.round(specificity, 4))

# Calculate the number of samples per class by summing the rows of the confusion matrix.
samples = np.sum(confMatrix, axis=1)

# Calculate the weights for each class as the proportion of samples in that class.
weights = samples / np.sum(confMatrix)

# Print the number of samples and weights for each class.
print("Samples:", samples)
print("Weights:", np.round(weights, 4))

# Calculate precision using weighted averaging: sum of precision per class multiplied by weights.
precision = np.sum(TP / (TP + FP) * weights)
# Calculate recall using weighted averaging: sum of recall per class multiplied by weights.
recall = np.sum(TP / (TP + FN) * weights)
# Calculate F1 score using weighted averaging: harmonic mean of weighted precision and recall.
f1 = 2 * precision * recall / (precision + recall)
# Calculate accuracy using weighted averaging: sum of TP and TN divided by the total sum of the matrix.
accuracy = np.sum((TP + TN) * weights) / np.sum(confMatrix)
# Calculate specificity using weighted averaging: sum of specificity per class multiplied by weights.
specificity = np.sum(TN / (TN + FP) * weights)

# Print the weighted-averaged metrics.
print("Weighted Precision:", np.round(precision, 4))
print("Weighted Recall:", np.round(recall, 4))
print("Weighted F1:", np.round(f1, 4))
print("Weighted Accuracy:", np.round(accuracy, 4))
print("Weighted Specificity:", np.round(specificity, 4))
