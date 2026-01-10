'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Jun 24th, 2025
# Last Modification Date: Jun 24th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import os  # For file and directory operations.
import pickle  # For saving and loading Python objects.
import pandas as pd  # For data manipulation and analysis.
import numpy as np  # For numerical operations.
from HMB_Summer_2025_Helpers import *

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"COVID-19 Radiography Database (FirstOrderFeatures-GLCM) Features.csv"
storageFolderName = r"COVID-19 Radiography Database (FirstOrderFeatures-GLCM) Features"

# Define the best model and scaler names.
bestModel = "MLP"
bestScaler = "Robust"
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
