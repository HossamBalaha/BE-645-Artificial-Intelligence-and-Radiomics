'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jul 14th, 2025
# Last Modification Date: Jul 14th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file and directory operations.
import tqdm  # For progress bar in loops.
import pickle  # For saving and loading Python objects.
import pandas as pd  # For data manipulation and analysis.
from HMB_Summer_2025_Helpers import *

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Train Features.csv"
testFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Test Features.csv"
storageFolderName = r"AdrenalMNIST3D (FirstOrderFeatures) V2"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLCM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLCM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLCM) V2"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLRLM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLRLM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLRLM) V2"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (GLSZM) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (GLSZM) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (GLSZM) V2"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (Shape) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (Shape) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (Shape) V2"

# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (All) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (All) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (All) V2"

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
  70,  # 70% of features
  80,  # 80% of features
  90,  # 90% of features
  100,  # 100% of features
  # You can also use:
  # 10,  # 10% of features
  # 20,  # 20% of features
  # 30,  # 30% of features
  # 40,  # 40% of features
  # 50,  # 50% of features
  # 60,  # 60% of features
]

# Create a list to store the performance metrics of each model and scaler combination.
history = []

# Iterate through each model and scaler combination.
for modelName in tqdm.tqdm(models, desc="Models"):
  for scalerName in tqdm.tqdm(scalers, desc="Scalers", leave=False):
    for fsTech in tqdm.tqdm(fsTechs, desc="Feature Selection Techniques", leave=False):
      for fsRatio in tqdm.tqdm(fsRatios, desc="Feature Selection Ratios", leave=False):
        try:
          # Call the function to perform machine learning classification.
          metrics, pltObject, objects = MachineLearningClassificationV2(
            os.path.join(baseDir, datasetFilename),  # Path to the dataset file.
            scalerName,  # Name of the scaler to be used.
            modelName,  # Name of the machine learning model to be used.
            fsTech,  # Feature selection technique to be applied.
            fsRatio,  # Ratio of features to be selected.
            testRatio=0.2,  # Ratio of the test data.
            testFilePath=os.path.join(baseDir, testFilename),  # Path to the test dataset file.
            targetColumn="Class",  # Name of the target column in the dataset.
            dropFirstColumn=True,  # Whether to drop the first column (usually an index or ID).
          )

          # UNCOMMENT THE FOLLOWING CODE TO PRINT THE METRICS WITH 4 DECIMAL PLACES.
          # Print the calculated metrics with 4 decimal places.
          # for key, value in metrics.items():
          #   print(f"{key}: {np.round(value, 4)}")

          # Create a pattern for the filename based on model name, scaler name, feature selection technique, and ratio.
          pattern = f"{modelName}_{scalerName}_{fsTech}_{fsRatio if (fsTech is not None) else None}"

          # Added to check if the plot object is not None before saving the plot.
          if (pltObject is not None):
            # Save the confusion matrix plot with a specific filename as a PNG image.
            pltObject.figure.savefig(
              os.path.join(storageFolderPath, f"{pattern}_CM.png"),
              bbox_inches="tight",  # Adjust the bounding box to fit the plot.
              dpi=720,  # Set the DPI for the saved image.
            )

            # pltObject.figure.show()  # Display the confusion matrix plot.
            pltObject.figure.clf()  # Clear the figure to free up memory.
            plt.close()  # Close the figure to free up memory.

          # Added to check if the objects are not None before saving them.
          if (objects is not None):
            # Save the trained model and scaler objects using pickle.
            with open(
              os.path.join(storageFolderPath, f"{pattern}.p"),
              "wb",  # Open the file in write-binary mode.
            ) as f:
              pickle.dump(objects, f)  # Save the model and scaler objects.

          # Append the model name and scaler name to the metrics dictionary.
          history.append(
            {
              "Model"                      : modelName,  # Name of the machine learning model.
              "Scaler"                     : scalerName,  # Name of the scaler used for preprocessing.
              "Feature Selection Technique": fsTech,  # Feature selection technique used.
              # Ratio of features selected, or "N/A" if no feature selection was applied.
              "Feature Selection Ratio"    : fsRatio if (fsTech is not None) else None,
              **metrics,  # Performance metrics returned by the classification function.
            }
          )
        except Exception as e:
          print(f"Error: {e}")

# Save the performance metrics in a CSV file for future reference.
df = pd.DataFrame(history)
df.to_csv(
  os.path.join(storageFolderPath, "Metrics History.csv"),
  index=False,
)

print("Done! The experiment has been completed successfully.")
