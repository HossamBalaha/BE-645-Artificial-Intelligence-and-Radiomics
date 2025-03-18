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
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

import tqdm, warnings, optuna, pickle
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

history = []


def OptunaObjectiveFunction(
  trial,  # Optuna trial object.
  # storagePath,  # The path to the storage directory.
  # filename,  # The name of the CSV file.
):
  # Get the parameters for the machine learning classification.
  modelName = trial.suggest_categorical("Model", models)
  scalerName = trial.suggest_categorical("Scaler", scalers)
  fsTech = trial.suggest_categorical("FS Tech", fsTechs)
  fsRatio = trial.suggest_categorical("FS Ratio", fsRatios)
  ovTech = trial.suggest_categorical("OV Tech", oversampleTechniques)

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

    # Return the weighted average of the metrics.
    return metrics["Weighted Average"]
  except Exception as e:
    # Uncomment the following line to print the error.
    # print(f"\nError: {e}")
    return 0


def TrainOptunaModel(
  nTrials=100,
  prefix="Optuna",
  storagePath="Data",
):
  # Create the study object.
  study = optuna.create_study(
    direction="maximize",  # To maximize the objective function.
    study_name=f"{prefix}_Study",  # The study name.
    storage=f"sqlite:///{storagePath}/{prefix}_Study.db",  # The database file.
    load_if_exists=True,  # To load the study if it exists.
    # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
    sampler=optuna.samplers.TPESampler(seed=42),  # Setting the sampler.
    # sampler=optuna.samplers.RandomSampler(seed=42),  # Setting the sampler.
  )

  # Create the objective function with the arguments.
  objectiveFunction = lambda trial: OptunaObjectiveFunction(
    trial,  # Optuna trial object.
  )
  # Optimize the objective function.
  study.optimize(
    objectiveFunction,  # Objective function.
    n_trials=nTrials,  # Number of trials.
    n_jobs=-1,  # To use all available CPUs for parallel execution.
    show_progress_bar=True,  # To show the progress bar.
  )

  # Save the performance metrics in a CSV file for future reference.
  df = pd.DataFrame(history)
  df.to_csv(
    os.path.join(storagePath, f"{filename.split('.')[0]} (using Optuna) Metrics History.csv"),
    index=False
  )

  # Store the trials in a dataframe.
  trials = study.trials_dataframe()
  trials.to_csv(
    os.path.join(storagePath, f"{prefix}_Optuna_Trials_History.csv"),
    index=False
  )

  # Get the best hyperparameters and the best value.
  bestParams = study.best_params
  bestValue = study.best_value
  print("Best Parameters:", bestParams)
  print("Best Value:", bestValue)

  # Save the best hyperparameters to a CSV file.
  bestParams = pd.DataFrame(bestParams, index=[0])
  bestParams.to_csv(
    os.path.join(storagePath, f"{prefix}_Optuna_Best_Params.csv"),
    index=False
  )

  # Save the study object.
  with open(os.path.join(storagePath, f"{prefix}_Optuna_Study.p"), "wb") as file:
    pickle.dump(study, file)

  return study, trials, bestParams, bestValue


# Train the Optuna model.
filenameNoExt = filename.split(".")[0]
study, trials, bestParams, bestValue = TrainOptunaModel(
  nTrials=500,
  prefix=f"Optuna_{filenameNoExt}",
  storagePath=storagePath,
)
