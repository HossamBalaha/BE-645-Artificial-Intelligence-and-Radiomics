# Author: Hossam Magdy Balaha
# Date: June 30th, 2024
# Permissions and Citation: Refer to the README file.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.ensemble import *
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

  results["TP"] = TP
  results["TN"] = TN
  results["FP"] = FP
  results["FN"] = FN

  return results


def CalculateAllMetricsBinary(cm):
  # Calculate TP, TN, FP, FN.
  TP = np.diag(cm)
  FP = np.sum(cm, axis=0) - TP
  FN = np.sum(cm, axis=1) - TP
  TN = np.sum(cm) - (TP + FP + FN)

  TN = TN[0]
  TP = TP[0]
  FP = FP[0]
  FN = FN[0]

  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = (TP + TN) / np.sum(cm)
  specificity = TN / (TN + FP)

  results = {
    "TP"         : TP,
    "TN"         : TN,
    "FP"         : FP,
    "FN"         : FN,
    "Precision"  : precision,
    "Recall"     : recall,
    "F1"         : f1,
    "Accuracy"   : accuracy,
    "Specificity": specificity
  }

  return results


def TrainEvaluateModel(model, xTrain, yTrain, xTest, yTest):
  model.fit(xTrain, yTrain)  # Train the model.
  predTest = model.predict(xTest)  # Evaluate the model.
  cm = confusion_matrix(yTest, predTest)  # Calculate the confusion matrix.
  if (len(np.unique(yTest)) == 2):
    metrics = CalculateAllMetricsBinary(cm)  # Calculate the metrics.
  else:
    metrics = CalculateAllMetrics(cm)  # Calculate the metrics.
  return metrics, cm


baseDir = r"MedMNISTv2"
trainPath = os.path.join(baseDir, "MedMNISTv2_Nodule_MNIST_3D_Train.csv")
testPath = os.path.join(baseDir, "MedMNISTv2_Nodule_MNIST_3D_Test.csv")
barPlotPath = os.path.join(baseDir, "MedMNISTv2_Nodule_MNIST_3D_Classes.png")
resultsStorePath = os.path.join(baseDir, "MedMNISTv2_Nodule_MNIST_3D_Test_Results.csv")
figStorePath = os.path.join(baseDir, "MedMNISTv2_Nodule_MNIST_3D_{s}_Test_CM.png")

# baseDir = r"MedMNISTv2"
# trainPath = os.path.join(baseDir, "MedMNISTv2_Organ_MNIST_3D_Train.csv")
# testPath = os.path.join(baseDir, "MedMNISTv2_Organ_MNIST_3D_Test.csv")
# barPlotPath = os.path.join(baseDir, "MedMNISTv2_Organ_MNIST_3D_Classes.png")
# resultsStorePath = os.path.join(baseDir, "MedMNISTv2_Organ_MNIST_3D_Train_Results.csv")
# figStorePath = os.path.join(baseDir, "MedMNISTv2_Organ_MNIST_3D_{s}_Test_CM.png")

# Load the data.
trainData = pd.read_csv(trainPath)
testData = pd.read_csv(testPath)

# Fill the missing values.
trainData = trainData.fillna(0)
testData = testData.fillna(0)

# Print the categories' information.
print(trainData["Class"].value_counts())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
trainData["Class"].value_counts().plot(kind="bar")
plt.tight_layout()
plt.subplot(1, 2, 2)
testData["Class"].value_counts().plot(kind="bar")
plt.tight_layout()
plt.savefig(barPlotPath, dpi=300, bbox_inches="tight")
plt.close()

xTrain = trainData.drop("Class", axis=1)
yTrain = trainData["Class"]
le = LabelEncoder()
yTrainEnc = le.fit_transform(yTrain)

xTest = testData.drop("Class", axis=1)
yTest = testData["Class"]
yTestEnc = le.transform(yTest)

models = [
  RandomForestClassifier(),
  DecisionTreeClassifier(),
  SVC(),
  KNeighborsClassifier(),
  MLPClassifier(),
  AdaBoostClassifier(),
  GradientBoostingClassifier(),
  ExtraTreesClassifier(),
  BaggingClassifier(),
]

history = []
for model in models:
  modelStr = model.__class__.__name__
  print("Working with:", modelStr)
  metrics, cm = TrainEvaluateModel(model, xTrain, yTrain, xTest, yTest)
  metrics["Model"] = modelStr
  history.append(metrics)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
  disp.plot()
  plt.savefig(figStorePath.format(s=modelStr), dpi=300, bbox_inches="tight")
  plt.close()

df = pd.DataFrame(history)
df.to_csv(resultsStorePath, index=False)
