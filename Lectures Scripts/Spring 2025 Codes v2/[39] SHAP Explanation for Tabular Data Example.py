'''
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Mar 25th, 2025
# Last Modification Date: Mar 25th, 2025
# Permissions and Citation: Refer to the README file.
'''

# Import necessary libraries.
import shap, warnings
import pandas as pd
import numpy as np
from HMB_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")

# Load the data from the specified CSV file.
filename = r"Merged_Filtered.csv"
storagePath = r"Data/3D_Features_20250303-124114"

# Get the parameters for the machine learning classification.
modelName = "ETs"
scalerName = "QT"
fsTech = None
fsRatio = 100
ovTech = "BorderlineSMOTE"

print("Experiment Setup:")
print("Model:", modelName)
print("Scaler:", scalerName)
print("Feature Selection Technique:", fsTech)
print("Feature Selection Ratio:", fsRatio)
print("Oversampling Technique:", ovTech)

metrics, storage, xTrain, xTest, yTrain, yTest, features = MachineLearningClassificationV4(
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

# Print the weighted average of the metrics.
print(f"Weighted Average: {metrics['Weighted Average']:0.4f}")

# Initialize SHAP explainer using the trained model and data.
# The explainer computes SHAP values to explain model predictions.
model = storage["Model"]
xTrainDF = pd.DataFrame(xTrain, columns=features)
xTestDF = pd.DataFrame(xTest, columns=features)
explainer = shap.Explainer(model.predict, xTrainDF)

# Compute SHAP values for the test set to explain model predictions.
shapValues = explainer(xTestDF)

# Display the shape of SHAP values.
print("SHAP values shape:", shapValues.shape)

# Make predictions on the test set.
testPred = model.predict(xTest)

# Index of the instance to explain.
instanceIndex = np.random.randint(0, xTestDF.shape[0])

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
  f"True Label: {yTest[instanceIndex]} and "
  f"Predicted Label: {testPred[instanceIndex]}"
)  # Set the title of the plot.
plt.tight_layout()  # Adjust layout to eliminate wasted space.
plt.savefig(f"{storagePath}/SHAP_Waterfall_Plot.png", dpi=300)  # Save the plot as an image.
plt.show()  # Render and display the plot.

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
plt.savefig(f"{storagePath}/SHAP_Scatter_Plot.png", dpi=300)  # Save the plot as an image.
plt.show()  # Render and display the plot.
