'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 18th, 2025
# Last Modification Date: Mar 18th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os, tqdm, warnings
import pandas as pd
import numpy as np
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
  None,  # No scaling
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

fsTechs = [
  None,  # No feature selection
  "PCA",  # Principal Component Analysis
  "LDA",  # Linear Discriminant Analysis
  "ANOVA",  # Analysis of Variance
  "MI",  # Mutual Information
  "RF",  # Random Forest Feature Importance
  "RFE",  # Recursive Feature Elimination
  "Chi2"  # Chi-Squared Feature Selection
]

fsRatios = [
  10,  # 10% of features
  20,  # 20% of features
  30,  # 30% of features
  40,  # 40% of features
  50,  # 50% of features
  60,  # 60% of features
  70,  # 70% of features
  80,  # 80% of features
  90,  # 90% of features
  100,  # 100% of features
]

oversampleTechniques = [
  None,  # No oversampling
  "SMOTE",  # Synthetic Minority Over-sampling Technique
  "ADASYN",  # Adaptive Synthetic Sampling Approach
  "BorderlineSMOTE",  # Borderline SMOTE
  "SVMSMOTE",  # Support Vector Machine SMOTE
]

np.random.shuffle(scalers)  # Shuffle the scalers for randomness.
np.random.shuffle(models)  # Shuffle the models for randomness.
np.random.shuffle(fsTechs)  # Shuffle the feature selection techniques for randomness.
np.random.shuffle(fsRatios)  # Shuffle the feature ratios for randomness.
np.random.shuffle(oversampleTechniques)  # Shuffle the oversampling techniques for randomness.

# Perform machine learning classification.
history = []
# NOTICE:
# THIS CODE WILL TAKE A LONG TIME TO RUN.
# BECAUSE IT WILL RUN 7 * 7 * 8 * 10 * 5 * 5 = 19600 TIMES.
# HENCE, WE NEED TO USE A BETTER APPROACH SUCH AS OPTUNA OR RANDOMIZED SEARCH.
for modelName in tqdm.tqdm(models, desc="Models"):
  for scalerName in tqdm.tqdm(scalers, desc="Scalers"):
    for fsTech in tqdm.tqdm(fsTechs, desc="FS Techniques"):
      for fsRatio in tqdm.tqdm(fsRatios, desc="FS Ratios"):
        for ovTech in tqdm.tqdm(oversampleTechniques, desc="Oversampling Techniques"):
          try:
            metrics = MachineLearningClassificationV3(
              storagePath,
              filename,
              modelName,
              scalerName,
              fsTech,
              fsRatio,
              ovTech,
              testRatio=0.2,
              targetColumn="Category",
            )

            # Append the model name and scaler name to the metrics dictionary.
            history.append(
              {
                "Model"   : modelName,
                "Scaler"  : scalerName,
                "FS Tech" : fsTech,
                "FS Ratio": fsRatio,
                "OV Tech" : ovTech,
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
  os.path.join(storagePath, f"{filename.split('.')[0]} (with FS/IMB) Metrics History.csv"),
  index=False
)
