'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 3rd, 2025
# Last Modification Date: Mar 18th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os, tqdm, warnings
import pandas as pd
from HMB_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")

# Load the data from the specified CSV file.
# filename = r"FOS_Filtered.csv"
# filename = r"GLCM_Filtered.csv"
# filename = r"GLRLM_Filtered.csv"
# filename = r"Shape_Filtered.csv"
filename = r"Merged_Filtered.csv"
storagePath = r"Data/3D_Features_20250303-124114"

scalers = [
  "Normalizer",  # Normalizer
  "Standard",  # Standard Scaler
  "MinMax",  # Min-Max Scaler
  "Robust",  # Robust Scaler
  "MaxAbs",  # Max Absolution Scaler
  "QT",  # Quantile Transformer
]

models = [
  "MLP",  # Multi-Layer Perceptron
  "RF",  # Random Forest
  "AB",  # Adaptive Boosting
  "KNN",  # K-Nearest Neighbors
  "DT",  # Decision Tree
  "ETs",  # Extra Trees Classifier
  "SGD",  # Stochastic Gradient Descent
  # You can also use (check GetMLClassificationModelObject function):
  # "SVC",  # Support Vector Classifier
  # "GNB",  # Gaussian Naive Bayes
  # "LR",  # Logistic Regression
  # "GB",  # Gradient Boosting Classifier
  # "Bagging",  # Bagging Classifier
  # "XGB",  # eXtreme Gradient Boosting
  # "LGBM",  # Light Gradient Boosting Machine
  # "Voting",  # Voting Classifier
  # "Stacking",  # Stacking Classifier
]

# Perform machine learning classification.
history = []
for modelName in tqdm.tqdm(models):
  for scalerName in tqdm.tqdm(scalers):
    try:
      metrics = MachineLearningClassification(
        storagePath,
        filename,
        scalerName,
        modelName,
        testRatio=0.2,
        targetColumn="Category",
      )

      # Append the model name and scaler name to the metrics dictionary.
      history.append(
        {
          "Model" : modelName,
          "Scaler": scalerName,
          **metrics,
        }
      )
    except Exception as e:
      # Uncomment the following line to print the error.
      # print(f"\nError: {e}")
      pass

# Save the performance metrics in a CSV file for future reference.
df = pd.DataFrame(history)
df.to_csv(
  os.path.join(storagePath, f"{filename.split('.')[0]} Metrics History.csv"),
  index=False
)
