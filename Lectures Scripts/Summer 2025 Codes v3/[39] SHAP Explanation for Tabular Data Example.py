'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 25th, 2025
# Last Modification Date: Jul 18th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import shap, warnings, pickle
import pandas as pd
import numpy as np
from HMB_Summer_2025_Helpers import *

# Access https://shap.readthedocs.io/en/latest/index.html
# for more information about SHAP and its visualization techniques.

# Load the data from the specified CSV file.
baseDir = "Data"  # Base directory.
datasetFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Train Features.csv"
testFilename = r"AdrenalMNIST3D (FirstOrderFeatures) Test Features.csv"
storageFolderName = r"AdrenalMNIST3D (FirstOrderFeatures) V3"

# # Load the data from the specified CSV file.
# baseDir = "Data"  # Base directory.
# datasetFilename = r"AdrenalMNIST3D (All) Train Features.csv"
# testFilename = r"AdrenalMNIST3D (All) Test Features.csv"
# storageFolderName = r"AdrenalMNIST3D (All) V3"

# Define the parameters for the experiment automatically.
optunaBestParamsFile = os.path.join(
  baseDir, storageFolderName, "Optuna Best Params.csv"
)  # Path to the file containing the best parameters from Optuna.
# Load the best parameters from the Optuna file.
optunaBestParamsDF = pd.read_csv(optunaBestParamsFile)
# Replace NaN values with "None".
optunaBestParamsDF.fillna("None", inplace=True)
# Extract the parameters from the DataFrame.
optunaBestParams = optunaBestParamsDF.iloc[0].to_dict()

# Print each parameter and its value.
print("Optuna Best Parameters:")
for key, value in optunaBestParams.items():
  print(f"{key}: {value}")

modelName = optunaBestParams["Model"]
scalerName = optunaBestParams["Scaler"] if (optunaBestParams["Scaler"] != "None") else None
fsTech = optunaBestParams["FS Tech"] if (optunaBestParams["FS Tech"] != "None") else None
fsRatio = optunaBestParams["FS Ratio"] if (optunaBestParams["FS Ratio"] != "None") else None
dataBalanceTech = optunaBestParams["DB Tech"] if (optunaBestParams["DB Tech"] != "None") else None

# Create a pattern for the filename based on model name, scaler name, feature selection technique, and ratio.
pattern = f"{modelName}_{scalerName}_{fsTech}_{fsRatio if (fsTech is not None) else None}_{dataBalanceTech}"
with open(
  os.path.join(baseDir, storageFolderName, f"{pattern}.p"),
  "rb",  # Open the file in write-binary mode.
) as f:
  # Load the storage dictionary from the file.
  objects = pickle.load(f)

testData = pd.read_csv(
  os.path.join(baseDir, testFilename),  # Read the test data from the specified CSV file.
  index_col=0,  # Use the first column as the index.
)

X = testData.drop(columns=["Class"])  # Drop the category column to get features.
y = testData["Class"]  # Extract the category column as the target variable.

# Use the columns after dropna as we did during training.
X = X[objects["CurrentColumns"]]
if (objects["Scaler"]):
  X = objects["Scaler"].transform(X)  # Normalize the features using the scaler.
  X = pd.DataFrame(X, columns=objects["CurrentColumns"])  # Convert the normalized features back to a DataFrame.
if (objects["FeatureSelector"]):
  X = objects["FeatureSelector"].transform(X)  # Apply feature selection to the normalized features.
  X = pd.DataFrame(X, columns=objects["SelectedFeatures"])  # Convert the selected features back to a DataFrame.

# Initialize SHAP explainer using the trained model and data.
# The explainer computes SHAP values to explain model predictions.
model = objects["Model"]
explainer = shap.Explainer(model.predict, X)

# Compute SHAP values for the test set to explain model predictions.
shapValues = explainer(X)

# Display the shape of SHAP values.
print("SHAP values shape:", shapValues.shape)

# Make predictions on the test set.
yPred = model.predict(X)
# Decode the predicted labels back to their original form.
yPredDecoded = objects["LabelEncoder"].inverse_transform(yPred)

# Index of the instance to explain.
instanceIndex = np.random.randint(0, X.shape[0])

# Class to explain.
categoryToExplain = 0  # Hyperplasia is the first class.

# Determine the number of records to visualize.
noOfRecords = 150

# Define the number of features to visualize.
noOfFeatures = 5

# Visualize the waterfall plot for an instance in the test set.
shap.plots.waterfall(
  shapValues[instanceIndex, :noOfFeatures],
  max_display=10,  # Show only the top 10 most important features.
  show=False,  # Prevent automatic display to allow customization.
)

plt.title(
  f"SHAP Waterfall Plot for Instance "
  f"{instanceIndex}\n"
  f"True Label: {y[instanceIndex]} and "
  f"Predicted Label: {yPredDecoded[instanceIndex]}"
)  # Set the title of the plot.
plt.tight_layout()  # Adjust layout to eliminate wasted space.
plt.savefig("Data/SHAP_Waterfall_Plot.png", dpi=720)  # Save the plot as an image.
plt.close()  # Close the plot to free up memory.

# Visualize the scatter plot for the test set.
shap.plots.scatter(
  shapValues[:, :noOfFeatures],
  show=False,  # Prevent automatic display to allow customization.
  # Color points by their SHAP values.
  color=shapValues[:, :noOfFeatures],
)

plt.title(
  f"SHAP Scatter Plot for the Test Set with "
  f"Reference Class {categoryToExplain}"
)  # Set the title of the plot.
plt.tight_layout()  # Adjust layout to eliminate wasted space.
plt.savefig("Data/SHAP_Scatter_Plot.png", dpi=720)  # Save the plot as an image.
plt.close()  # Close the plot to free up memory.

# Visualize the summary plot for the test set.
shap.summary_plot(
  shapValues[:, :noOfFeatures],
  show=False,  # Prevent automatic display to allow customization.
)

plt.title(
  f"SHAP Summary Plot for the Test Set with "
  f"Reference Class {categoryToExplain}"
)  # Set the title of the plot.
plt.tight_layout()  # Adjust layout to eliminate wasted space.
plt.savefig("Data/SHAP_Summary_Plot.png", dpi=720)  # Save the plot as an image.
plt.close()  # Close the plot to free up memory.

# Visualize the force plot for an instance in the test set.
shap.plots.force(
  shapValues[instanceIndex, :noOfFeatures],  # SHAP values for the instance.
  matplotlib=True,  # Use Matplotlib for plotting.
  show=False,  # Prevent automatic display to allow customization.
)

plt.title(
  f"SHAP Force Plot for Instance "
  f"{instanceIndex}\n"
  f"True Label: {y[instanceIndex]} and "
  f"Predicted Label: {yPredDecoded[instanceIndex]}"
)  # Set the title of the plot.
plt.tight_layout()  # Adjust layout to eliminate wasted space.
plt.savefig("Data/SHAP_Force_Plot.png", dpi=720)  # Save the plot as an image.
plt.close()  # Close the plot to free up memory.
