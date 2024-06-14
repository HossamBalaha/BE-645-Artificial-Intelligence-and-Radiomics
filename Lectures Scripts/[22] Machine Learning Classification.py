# Author: Hossam Magdy Balaha
# Date: June 13th, 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


def CalculateAllMetrics(cm):
  # Calculate TP, TN, FP, FN.
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = np.sum(cm) - (TP + FP + FN)

  results = {}

  # Using macro averaging.
  precision = np.mean(TP / (TP + FP))
  recall = np.mean(TP / (TP + FN))
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.mean(TN / (TN + FP))

  results["Macro Precision"] = precision
  results["Macro Recall"] = recall
  results["Macro F1"] = f1
  results["Macro Accuracy"] = accuracy
  results["Macro Specificity"] = specificity

  # Using micro averaging.
  precision = np.sum(TP) / np.sum(TP + FP)
  recall = np.sum(TP) / np.sum(TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.sum(TN) / np.sum(TN + FP)

  results["Micro Precision"] = precision
  results["Micro Recall"] = recall
  results["Micro F1"] = f1
  results["Micro Accuracy"] = accuracy
  results["Micro Specificity"] = specificity

  # Using weighted averaging.
  samples = np.sum(cm, axis=1)
  weights = samples / np.sum(cm)

  precision = np.sum(TP / (TP + FP) * weights)
  recall = np.sum(TP / (TP + FN) * weights)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = np.sum(TP) / np.sum(cm)
  specificity = np.sum(TN / (TN + FP) * weights)

  results["Weighted Precision"] = precision
  results["Weighted Recall"] = recall
  results["Weighted F1"] = f1
  results["Weighted Accuracy"] = accuracy
  results["Weighted Specificity"] = specificity

  return results


# Load the data.
data = pd.read_csv("COVID-19 Radiography Database 2D.csv")

# Drop the null and empty values.
data = data.dropna()

# Split the data into features and target.
testRatio = 0.2

X = data.drop("Class", axis=1)
y = data["Class"]
le = LabelEncoder()
yEnc = le.fit_transform(y)

xTrain, xTest, yTrain, yTest = train_test_split(
  X, yEnc,
  test_size=testRatio,
  random_state=42,
  stratify=y
)

# Train the model.
model = RandomForestClassifier()
model.fit(xTrain, yTrain)

# Evaluate the model.
predTest = model.predict(xTest)

# Calculate the confusion matrix.
cm = confusion_matrix(yTest, predTest)

metrics = CalculateAllMetrics(cm)
for key, value in metrics.items():
  print(f"{key}: {value}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot()
plt.show()
plt.close()
