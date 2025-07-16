'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 18th, 2025
# Last Modification Date: Jul 15th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file path operations.

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OMP_NUM_THREADS'] = "1"

from HMB_Summer_2025_Helpers import *

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Train Features.csv"
testFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Test Features.csv"
storageFolderName = r"AdrenalMNIST3D (FirstOrderFeatures) V3"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLCM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLCM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLCM) V3"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLRLM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLRLM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLRLM) V3"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLSZM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLSZM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLSZM) V3"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (Shape) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (Shape) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (Shape) V3"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (All) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (All) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (All) V3"

# Create the storage folder path if it does not exist.
storageFolderPath = os.path.join(baseDir, storageFolderName)
os.makedirs(
  storageFolderPath,
  exist_ok=True,  # Create the directory if it does not exist.
)

scalers = [
  None,  # No scaling (Added to test the performance of the models without scaling)
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
  "RF",  # Random Forest Feature Importance
  "RFE",  # Recursive Feature Elimination
  # You can also use:
  # "LDA",  # Linear Discriminant Analysis
  # "MI",  # Mutual Information
  # "Chi2",  # Chi-Squared Feature Selection
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

dataBalanceTechniques = [
  None,  # No data balancing (Added to test the performance of the models without balancing)
  "SMOTE",  # Synthetic Minority Over-sampling Technique
  "ADASYN",  # Adaptive Synthetic Sampling Approach
  "BorderlineSMOTE",  # Borderline SMOTE
  "SVMSMOTE",  # Support Vector Machine SMOTE
  "RandomOverSampler",  # Random Over-Sampling
  "RandomUnderSampler",  # Random Under-Sampling
  "NearMiss",  # Near Miss Sampling
  "ClusterCentroids",  # Cluster Centroids
]

if __name__ == "__main__":
  filenameNoExt = datasetFilename.split(".")[0]  # Get the filename without extension.
  numTrials = 150  # Number of trials for hyperparameter tuning.

  tuneObj = OptunaTuning(
    scalers=scalers,  # List of scalers to be used in the tuning process.
    models=models,  # List of machine learning models to be used in the tuning process.
    fsTechs=fsTechs,  # List of feature selection techniques to be used in the tuning process.
    fsRatios=fsRatios,  # List of feature selection ratios to be used in the tuning process.
    dataBalanceTechniques=dataBalanceTechniques,  # List of data balancing techniques to be used in the tuning process.
    baseDir=baseDir,  # Base directory where the dataset is stored.
    datasetFilename=datasetFilename,  # Name of the dataset file.
    storageFolderPath=storageFolderPath,  # Path to the folder where results will be stored.
    testFilename=testFilename,  # Name of the test dataset file.
    testRatio=0.2,  # Ratio of the test data.
    numTrials=numTrials,  # Number of trials for hyperparameter tuning.
  )

  # Run the tuning process.
  tuneObj.Tune()
