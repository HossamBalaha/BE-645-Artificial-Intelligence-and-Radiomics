"""
========================================================================
        ╦ ╦┌─┐┌─┐┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌─┐┌┬┐┬ ┬  ╔╗ ┌─┐┬  ┌─┐┬ ┬┌─┐
        ╠═╣│ │└─┐└─┐├─┤│││  ║║║├─┤│ ┬ ││└┬┘  ╠╩╗├─┤│  ├─┤├─┤├─┤
        ╩ ╩└─┘└─┘└─┘┴ ┴┴ ┴  ╩ ╩┴ ┴└─┘─┴┘ ┴   ╚═╝┴ ┴┴─┘┴ ┴┴ ┴┴ ┴
========================================================================
# Author: Hossam Magdy Balaha
# Initial Creation Date: Apr 8th, 2025
# Last Modification Date: Apr 8th, 2025
# Permissions and Citation: Refer to the README file.
"""

# Import necessary libraries.
import os, warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import *
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from HMB_Helpers import *

# Ignore warnings.
warnings.filterwarnings("ignore")

# Check if GPU is available and set memory growth.
gpus = tf.config.experimental.list_physical_devices("GPU")
print("Available GPUs:", gpus)

# Define paths and constants.
# Replace with the actual path to your dataset.
datasetDir = r"..\..\Datasets\Dataset_BUSI_with_GT_Classes"
imageSize = (128, 128)  # Input size for MobileNetV2.
batchSize = 32  # Batch size for training, validation, and testing.
numEpochs = 1000  # Number of epochs for training.

# Load dataset file paths and labels.
filePaths = []  # List to store image file paths.
labels = []  # List to store corresponding labels.

# Iterate through the dataset directory to collect file paths and labels.
for classDir in os.listdir(datasetDir):
  classPath = os.path.join(datasetDir, classDir)
  if os.path.isdir(classPath):  # Check if the path is a directory.
    for imgFile in os.listdir(classPath):
      filePaths.append(os.path.join(classPath, imgFile))  # Add file path.
      labels.append(classDir)  # Add corresponding label.

# Convert lists to numpy arrays for processing.
labelEncoder = LabelEncoder()
encodedLabels = labelEncoder.fit_transform(labels)  # Encode labels as integers.

# Split the data into training, validation, and testing sets.
trainFiles, testFiles, trainLabels, testLabels = train_test_split(
  filePaths, encodedLabels, test_size=0.2, random_state=42
)  # Split 80% for training and 20% for testing.
trainFiles, valFiles, trainLabels, valLabels = train_test_split(
  trainFiles, trainLabels, test_size=0.25, random_state=42
)  # Further split training data into 60% training and 20% validation.


# Create a function to load and preprocess images.
def LoadImage(filePath, label):
  img = tf.io.read_file(filePath)  # Read the image file.
  img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG image with 3 channels.
  img = tf.image.resize(img, imageSize)  # Resize image to (224, 224).
  img = img / 255.0  # Normalize pixel values to [0, 1].
  return img, label  # Return the preprocessed image and its label.


# Create TensorFlow datasets for training, validation, and testing.
trainDataset = tf.data.Dataset.from_tensor_slices((trainFiles, trainLabels))
trainDataset = trainDataset.map(LoadImage, num_parallel_calls=tf.data.AUTOTUNE)  # Apply preprocessing.
trainDataset = trainDataset.shuffle(buffer_size=1000).batch(batchSize).prefetch(tf.data.AUTOTUNE)

valDataset = tf.data.Dataset.from_tensor_slices((valFiles, valLabels))
valDataset = valDataset.map(LoadImage, num_parallel_calls=tf.data.AUTOTUNE)  # Apply preprocessing.
valDataset = valDataset.batch(batchSize).prefetch(tf.data.AUTOTUNE)

testDataset = tf.data.Dataset.from_tensor_slices((testFiles, testLabels))
testDataset = testDataset.map(LoadImage, num_parallel_calls=tf.data.AUTOTUNE)  # Apply preprocessing.
testDataset = testDataset.batch(batchSize).prefetch(tf.data.AUTOTUNE)

# Load the pre-trained model without the top layers.
baseModel = DenseNet201(weights="imagenet", include_top=False, input_shape=(*imageSize, 3))

# Freeze the base model layers to prevent them from being trained initially.
baseModel.trainable = False

# Add custom classification layers on top of the base model.
x = baseModel.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions to 1D.
x = Dense(128, activation="relu")(x)  # Add a dense layer with ReLU activation.
x = Dropout(0.5)(x)  # Add dropout for regularization.
predictions = Dense(len(labelEncoder.classes_), activation="softmax")(x)  # Output layer with 3 classes.

# Create the final model.
model = Model(inputs=baseModel.input, outputs=predictions)

# Create callbacks for early stopping and model checkpointing.
earlyStopping = EarlyStopping(
  monitor="val_loss",  # Monitor validation loss.
  patience=25,  # Stop training if no improvement for number of epochs.
  restore_best_weights=True  # Restore the best weights.
)

modelCheckpoint = ModelCheckpoint(
  filepath=r"Data/BestCNNModel.h5",  # Save the best model.
  monitor="val_loss",  # Monitor validation loss.
  save_best_only=True,  # Save only the best model.
  mode="min",  # Save the model with minimum validation loss.
  verbose=1  # Print messages when saving the model.
)

# Compile the model with Adam optimizer and categorical crossentropy loss.
model.compile(
  optimizer=Adam(learning_rate=0.0001),  # Use a small learning rate for transfer learning.
  loss="sparse_categorical_crossentropy",  # Use sparse categorical crossentropy for multi-class classification.
  metrics=["accuracy"],  # Monitor accuracy during training.
)

# Train the model on the training data.
history = model.fit(
  trainDataset,
  epochs=numEpochs,  # Train for a specified number of epochs.
  validation_data=valDataset,  # Use validation data for monitoring.
  callbacks=[earlyStopping, modelCheckpoint]  # Use callbacks for early stopping and checkpointing.
)

# Restore the best model weights from the checkpoint.
model.load_weights(r"Data/BestCNNModel.h5")

# Evaluate the model on the test set.
loss, accuracy = model.evaluate(testDataset)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Find the test predictions.
testPredictions = model.predict(testDataset)
# Convert predictions to class labels.
testPredictions = np.argmax(testPredictions, axis=1)
# Convert test labels to class labels.
testLabels = np.array(testLabels)
# Create a confusion matrix.
cm = confusion_matrix(testLabels, testPredictions)
# Calculate performance metrics using the custom PerformanceMetrics function.
metrics = PerformanceMetrics(cm)
# Print the confusion matrix and performance metrics.
print("Confusion Matrix:\n", cm)
for metric, value in metrics.items():
  print(f"{metric}: {value}")

# Plot training history for accuracy.
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
