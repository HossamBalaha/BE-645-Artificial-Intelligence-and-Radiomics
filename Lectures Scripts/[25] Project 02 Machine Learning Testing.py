'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file and directory operations.
import pickle  # For saving and loading Python objects.
import pandas as pd  # For data manipulation and analysis.
import numpy as np  # For numerical operations.
from HMB_Spring_2026_Helpers import *

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"COVID-19 Radiography Database (FirstOrderFeatures) Features.csv"
storageFolderName = r"COVID-19 Radiography Database (FirstOrderFeatures) Features"
resultsFilename = "Metrics History.csv"

# Read the dataset into a DataFrame.
df = pd.read_csv(os.path.join(baseDir, storageFolderName, resultsFilename))
# Find the index of the top-1 model based on the "Weighted Average" column.
top1Index = df["Weighted Average"].idxmax()
record = df.iloc[top1Index]
print(f"Top-1 Model Index: {top1Index}")
print(f"Top-1 Model Details:\n{record}")

# Define the best model and scaler names.
bestModel = record["Model"]
bestScaler = record["Scaler"]
categoryColumnName = "Class"
dropFirstColumn = True

# Get the pickle path based on the model and scaler names.
picklePath = os.path.join(
  baseDir,
  storageFolderName,
  f"{bestModel}_{bestScaler}.p",
)

# Load the best model and scaler from the pickle file.
with open(picklePath, "rb") as file:
  objects = pickle.load(file)

# Extract the model, scaler, and label encoder from the loaded objects.
model = objects["Model"]
scaler = objects["Scaler"]
labelEncoder = objects["LabelEncoder"]

# Load the dataset into a DataFrame.
df = pd.read_csv(os.path.join(baseDir, datasetFilename))

# Get a random sample from the dataset.
rndIndex = np.random.randint(0, len(df))
sample = df.iloc[rndIndex]
X = sample.drop(categoryColumnName).values.reshape(1, -1)  # Features.
y = sample[categoryColumnName]  # Target variable.

# Check if the first column should be dropped.
if (dropFirstColumn):
  # Drop the first column if specified.
  X = X[:, 1:]

# Scale the features using the scaler.
xScaled = scaler.transform(X)

# Predict the class using the model.
yPred = model.predict(xScaled)

# Decode the predicted class label.
yPredDecoded = labelEncoder.inverse_transform(yPred)

print(f"Sample Index: {rndIndex}")
print(f"Sample Features:\n{X}")
print(f"Sample Target: {y}")
print(f"Predicted Class: {yPredDecoded[0]}")
