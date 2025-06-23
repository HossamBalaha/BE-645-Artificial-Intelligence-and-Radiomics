'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 13th, 2024
# Last Modification Date: Jun 23rd, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file and directory operations.
import tqdm  # For progress bar in loops.
import pandas as pd  # For data manipulation and analysis.
from HMB_Summer_2025_Helpers import *

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"COVID-19 Radiography Database (FirstOrderFeatures) Features.csv"
storageFolderName = r"COVID-19 Radiography Database (FirstOrderFeatures) Features"

# Load the data from the specified CSV file.
# baseDir = "Data"  # Base directory.
# datasetFilename = r"COVID-19 Radiography Database (GLCM) Features.csv"
# storageFolderName = r"COVID-19 Radiography Database (GLCM) Features"

# Load the data from the specified CSV file.
# baseDir = "Data"  # Base directory.
# datasetFilename = r"COVID-19 Radiography Database (GLRLM) Features.csv"
# storageFolderName = r"COVID-19 Radiography Database (GLRLM) Features"

# Load the data from the specified CSV file.
# baseDir = "Data"  # Base directory.
# datasetFilename = r"COVID-19 Radiography Database (GLSZM) Features.csv"
# storageFolderName = r"COVID-19 Radiography Database (GLSZM) Features"

# Create the storage folder path if it does not exist.
storageFolderPath = os.path.join(baseDir, storageFolderName)
os.makedirs(
  storageFolderPath,
  exist_ok=True,  # Create the directory if it does not exist.
)

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

# Create a list to store the performance metrics of each model and scaler combination.
history = []

# Iterate through each model and scaler combination.
for modelName in tqdm.tqdm(models, desc="Models"):
  for scalerName in tqdm.tqdm(scalers, desc="Scalers", leave=False):
    try:
      # Call the function to perform machine learning classification.
      metrics, pltObject = MachineLearningClassificationV1(
        os.path.join(baseDir, datasetFilename),  # Path to the dataset file.
        scalerName,  # Name of the scaler to be used.
        modelName,  # Name of the machine learning model to be used.
      )

      # Save the confusion matrix plot with a specific filename as a PNG image.
      pltObject.figure.savefig(
        os.path.join(storageFolderPath, f"{scalerName} {modelName} CM.png"),
        bbox_inches="tight",  # Adjust the bounding box to fit the plot.
        dpi=720,  # Set the DPI for the saved image.
      )

      # pltObject.figure.show()  # Display the confusion matrix plot.
      pltObject.figure.clf()  # Clear the figure to free up memory.
      plt.close()  # Close the figure to free up memory.

      # Append the model name and scaler name to the metrics dictionary.
      history.append(
        {
          "Model" : modelName,  # Name of the machine learning model.
          "Scaler": scalerName,  # Name of the scaler used for preprocessing.
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

print("Done! The metrics history has been saved successfully.")
