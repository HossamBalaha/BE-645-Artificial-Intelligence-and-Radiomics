'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 13th, 2024
# Last Modification Date: Feb 9th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os, tqdm, warnings
import pandas as pd
from HMB_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")

# Load the data from the specified CSV file.
# filename = r"Records_1000_[0]_[1]_True_True.csv"
# # filename = r"Records_1000_[0, 90, 45, 135]_[1, 2, 3]_True_True.csv"
# storagePath = r"Data/COVID-19 Radiography Database 2D"

# Load the data from the specified CSV file.
filename = r"Records_9999_[0]_[1]_True_True.csv"
storagePath = r"Data/BUSI 2D"

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

# Perform machine learning classification using different scalers and models.
history = []
for modelName in tqdm.tqdm(models):
  for scalerName in tqdm.tqdm(scalers):
    try:
      metrics = MachineLearningClassification(
        storagePath,
        filename,
        scalerName,
        modelName,
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
      print(f"Error: {e}")

# Save the performance metrics in a CSV file for future reference.
df = pd.DataFrame(history)
df.to_csv(
  os.path.join(storagePath, f"{filename.split('.')[0]} Metrics History.csv"),
  index=False
)
