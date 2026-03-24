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
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import Adam
from HMB_Spring_2026_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")
# Print TensorFlow version for transparency.
print("TensorFlow Version:", tf.__version__)
# Print number of visible GPUs.
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

# Define paths and constants.
# Replace with the actual path to your dataset.
datasetDir = r"..\..\Datasets\Dataset_BUSI_with_GT_Classes"

# Define the network input image shape (H, W, C).
inputShape = (256, 256, 3)
# Training batch size.
batchSize = 64
# Number of epochs to train.
epochs = 200
# Define the output directory for saving models and logs.
outputDir = "History/Pretrained Models Fine-Tuning Training"
# Create the output directory if it does not exist.
os.makedirs(outputDir, exist_ok=True)

# Load dataset file paths and labels.
filePaths = []  # List to store image file paths.
labels = []  # List to store corresponding labels.

# Iterate through the dataset directory to collect file paths and labels.
for classDir in os.listdir(datasetDir):
  classPath = os.path.join(datasetDir, classDir)
  if (os.path.isdir(classPath)):  # Check if the path is a directory.
    for imgFile in os.listdir(classPath):
      filePaths.append(os.path.join(classPath, imgFile))  # Add file path.
      labels.append(classDir)  # Add corresponding label.

# Create a DataFrame to organize image paths and labels for easier handling.
dataFrame = pd.DataFrame({"image_path": filePaths, "label": labels})
print(f"Created DataFrame with {len(dataFrame)} rows.")
print("DataFrame head:\n", dataFrame.head())
noOfClasses = len(dataFrame["label"].unique())
print(f"Number of unique classes: {noOfClasses}")

# Train-test split of the dataset using sklearn's train_test_split function.
trainDF, testDF = train_test_split(
  dataFrame,  # DataFrame containing image paths and labels.
  test_size=0.2,  # Use 20% of the data for testing and 80% for training.
  random_state=42,  # Set a random state for reproducibility of the split.
  stratify=dataFrame["label"],  # Stratify the split based on the labels to maintain class distribution in both sets.
)
# Further split the training set into training and validation sets.
trainDF, valDF = train_test_split(
  trainDF,  # DataFrame containing training image paths and labels.
  test_size=0.25,  # Use 25% of the training data for validation (which is 20% of the original data).
  random_state=42,  # Set a random state for reproducibility of the split.
  stratify=trainDF["label"],  # Stratify the split based on the labels to maintain class distribution in both sets.
)
print(f"Training set size: {len(trainDF)}")
print(f"Validation set size: {len(valDF)}")
print(f"Testing set size: {len(testDF)}")

# Create a data generator for training with simple rescaling.
trainDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1].
  rotation_range=10,  # Randomly rotate images by up to 10 degrees for augmentation.
  width_shift_range=0.15,  # Randomly shift images horizontally by up to 15% of the width for augmentation.
  height_shift_range=0.15,  # Randomly shift images vertically by up to 15% of the height for augmentation.
  shear_range=0.15,  # Randomly apply shearing transformations to images for augmentation.
  zoom_range=0.15,  # Randomly zoom in on images by up to 15% for augmentation.
  horizontal_flip=True,  # Randomly flip images horizontally for augmentation.
  vertical_flip=False,  # Do not flip images vertically as it may not be appropriate for histopathological images.
  fill_mode="nearest",  # Fill in newly created pixels after transformations with the nearest pixel values.
)

# Create a generator that reads images and labels from the training DataFrame.
trainGen = trainDataGen.flow_from_dataframe(
  trainDF,  # DataFrame containing training image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize images to match model input size.
  batch_size=batchSize,  # Batch size for training.
  class_mode="categorical",  # Use one-hot encoded labels for training.
  shuffle=True,  # Shuffle training data to improve generalization.
)

# Create a data generator for validation with simple rescaling.
valDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1].
)

# Create a generator that reads images and labels from the validation DataFrame.
valGen = valDataGen.flow_from_dataframe(
  valDF,  # DataFrame containing validation image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize to model input size.
  batch_size=batchSize,  # Batch size for validation.
  class_mode="categorical",  # Use one-hot encoded labels for validation.
  shuffle=True,  # Shuffle validation data (optional, can be False if order matters for metrics).
)

# Create a data generator for testing with simple rescaling.
testDataGen = ImageDataGenerator(
  rescale=1.0 / 255.0,  # Preserve ordering for metrics.
)

# Instantiate the test generator to evaluate final performance.
testGen = testDataGen.flow_from_dataframe(
  testDF,  # DataFrame containing test image paths and labels.
  x_col="image_path",  # Column in DataFrame that contains image file paths.
  y_col="label",  # Column in DataFrame that contains class labels.
  target_size=inputShape[:2],  # Resize to model input size.
  batch_size=batchSize,  # Batch size for testing (can be larger since no backprop).
  class_mode="categorical",  # Use one-hot encoded labels for evaluation.
  shuffle=False,  # Do not shuffle test data to maintain order for metrics like confusion matrix.
)

for text, gen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  # Print the number of samples and classes in each generator for verification.
  print(f"Generator '{gen.directory}' has {gen.samples} samples and {len(gen.class_indices)} classes.")
  # Display a sample figure of the first batch of training images and their corresponding labels.
  xBatch, yBatch = next(gen)  # Get the first batch of images and labels from the training generator.
  plt.figure(figsize=(12, 12))  # Create a figure with a specific size for better visibility.
  # Loop through the first 16 images in the batch (or fewer if batch size is smaller).
  for i in range(min(16, len(xBatch))):
    plt.subplot(4, 4, i + 1)  # Create a subplot for each image in a 4x4 grid.
    plt.imshow(xBatch[i])  # Display the image (already rescaled to [0, 1]).
    # Show the class label as the title (convert one-hot to class index).
    clsIdx = np.argmax(yBatch[i])  # Get the class index from the one-hot encoded label.
    # Get the class name from the index.
    clsName = list(gen.class_indices.keys())[list(gen.class_indices.values()).index(clsIdx)]
    # Set the title of the subplot to the class name.
    plt.title(clsName + f" ({clsIdx})")  # Show class name and index for clarity.
    plt.axis("off")  # Hide axes for cleaner visualization.
  plt.tight_layout()  # Adjust layout to prevent overlap.
  # Save the sample batch figure to the output directory.
  plt.savefig(f"{outputDir}/{text}_SampleBatch.png")  # Save the figure to the History folder.
  # plt.show()  # Uncomment this line if you want to see the plot during execution.
  plt.close()  # Close the figure to free memory.

baseModel = MobileNetV2(
  # Exclude default classification head; we'll add a custom head.
  include_top=False,
  # Initialize with ImageNet weights.
  weights="imagenet",
  # Input image shape for the model.
  input_shape=inputShape,
)

model = PretrainedCNN(
  baseModel,
  inputShape,  # Input shape for the model (height, width, channels).
  optimizer=Adam(),  # Use Adam optimizer for training.
  noOfClasses=noOfClasses,  # Set the number of output classes.
  verbose=1,  # Print model summary when creating the model.
)

# Train the model with several useful callbacks.
history = model.fit(
  trainGen,  # Training data generator.
  epochs=epochs,  # Total number of epochs.
  batch_size=batchSize,  # Batch size (note generators already yield batches).
  validation_data=valGen,  # Validation data generator.
  callbacks=[
    # Save the best model (by validation categorical accuracy).
    ModelCheckpoint(
      f"{outputDir}/Model.keras",  # Filepath to save the best model.
      save_best_only=True,  # Only save the model if validation accuracy improves.
      save_weights_only=False,  # Save the entire model (architecture + weights).
      monitor="val_loss",  # Metric to monitor for improvement.
      verbose=1,  # Print messages when a new best model is saved.
    ),
    # Stop training early if validation stops improving.
    EarlyStopping(patience=50),
    # Record training history to CSV for later analysis.
    CSVLogger(f"{outputDir}/Log.log"),
    # Reduce learning rate when a monitored metric plateaus.
    ReduceLROnPlateau(factor=0.5, patience=10),
    # TensorBoard callback to log training for visualization.
    TensorBoard(log_dir=f"{outputDir}/TB/Logs", histogram_freq=1),
  ],
  verbose=2,  # Print progress during training (1 for progress bar, 2 for one line per epoch, 0 for silent).
)

# Load the best model saved during training.
model.load_weights(f"{outputDir}/Model.keras")

# Evaluate the trained model on training, validation, and test splits.
for name, dataGen in [("Training", trainGen), ("Validation", valGen), ("Testing", testGen)]:
  # Print which split is being evaluated.
  print(f"{name} Evaluation:")
  # Evaluate generator and capture results.
  result = model.evaluate(dataGen, batch_size=batchSize, verbose=0)
  # Print the common metrics in the order returned by model.evaluate.
  print("Loss:", result[0])
  print("Accuracy:", result[1])
  print("Precision:", result[2])
  print("Recall:", result[3])
  print("AUC:", result[4])
  print("TP:", result[5])
  print("TN:", result[6])
  print("FP:", result[7])
  print("FN:", result[8])

# Plot training and validation loss and accuracy curves across epochs.
plt.figure()
plt.subplot(2, 1, 1)
# Plot training and validation loss.
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2, 1, 2)
# Plot training and validation categorical accuracy.
plt.plot(history.history["categorical_accuracy"], label="Training Accuracy")
plt.plot(history.history["val_categorical_accuracy"], label="Validation Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
# Save the figure to the History folder for later review.
plt.savefig(f"{outputDir}/History.png")
# Display the plot interactively.
# plt.show()  # Uncomment this line if you want to see the plot during execution.
plt.close()

# Generate predictions on the test set using the trained model.
yCatPred = model.predict(testGen, batch_size=batchSize, verbose=0)
# Convert the one-hot predictions to class indices.
yPred = np.argmax(yCatPred, axis=1)
# Get true class indices from the test generator.
yTrue = testGen.classes
# Compute the confusion matrix between true and predicted labels.
cm = confusion_matrix(yTrue, yPred)
# Create a display object and plot the confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testGen.class_indices)
# Show the confusion matrix plot.
disp.plot()
# Rotate x-axis labels for better readability.
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent overlap of labels and titles.
# Save the confusion matrix figure to the History folder.
plt.savefig(f"{outputDir}/ConfusionMatrix.png")
# plt.show()  # Uncomment this line if you want to see the plot during execution.
plt.close()

# Calculate all metrics.
print("Confusion Matrix:")
cm = confusion_matrix(yTrue, yPred)
print("Performance Metrics:")
results = CalculatePerformanceMetrics(cm)
for key, value in results.items():
  print(f"{key}:", value)
