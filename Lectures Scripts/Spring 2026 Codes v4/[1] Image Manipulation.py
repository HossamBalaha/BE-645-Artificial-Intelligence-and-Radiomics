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
import cv2  # For image processing tasks.
import matplotlib.pyplot as plt  # For displaying images.

# Define the paths to the input image and segmentation mask.
caseImgPath = r"Data/Sample Liver Image.bmp"  # Path to the liver image.
caseSegPath = r"Data/Sample Liver Segmentation.bmp"  # Path to the liver segmentation mask.

# Load the images in grayscale mode.
caseImg = cv2.imread(caseImgPath, cv2.IMREAD_GRAYSCALE)  # Load the liver image.
caseSeg = cv2.imread(caseSegPath, cv2.IMREAD_GRAYSCALE)  # Load the segmentation mask.

# Get the shape (dimensions) of the images.
caseImgShape = caseImg.shape  # Shape of the liver image.
caseSegShape = caseSeg.shape  # Shape of the segmentation mask.

# Print the shapes of the images.
print("Image Shape: ", caseImgShape)  # Print the shape of the liver image.
print("Segmentation Shape: ", caseSegShape)  # Print the shape of the segmentation mask.

# Extract the Region of Interest (ROI) using the segmentation mask.
# Apply bitwise AND operation to extract the ROI.
roi = cv2.bitwise_and(
  caseImg,  # Image.
  caseSeg,  # Segmentation mask.
)

# Save the extracted ROI to a new image file.
cv2.imwrite(
  caseImgPath.replace("Image.bmp", "ROI.bmp"),  # Path to save the ROI image.
  roi,  # Image to save as the ROI.
)

# Crop the ROI to remove unnecessary background.
x, y, w, h = cv2.boundingRect(roi)  # Get the bounding box coordinates of the ROI.
# x: x-coordinate of the top-left corner of the bounding box.
# y: y-coordinate of the top-left corner of the bounding box.
# w: width of the bounding box.
# h: height of the bounding box.
cropped = roi[y:y + h, x:x + w]  # Crop the ROI using the bounding box coordinates.

# Display the images using matplotlib.
plt.figure(figsize=(10, 3))  # Create a new figure.

# Display the original liver image.
plt.subplot(1, 4, 1)  # Create a subplot in the first position.
plt.imshow(caseImg, cmap="gray")  # Display the liver image in grayscale.
plt.title("Image")  # Set the title of the subplot.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.

# Display the segmentation mask.
plt.subplot(1, 4, 2)  # Create a subplot in the second position.
plt.imshow(caseSeg, cmap="gray")  # Display the segmentation mask in grayscale.
plt.title("Segmentation")  # Set the title of the subplot.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.

# Display the extracted ROI.
plt.subplot(1, 4, 3)  # Create a subplot in the third position.
plt.imshow(roi, cmap="gray")  # Display the ROI in grayscale.
plt.title("ROI")  # Set the title of the subplot.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.

# Display the cropped ROI.
plt.subplot(1, 4, 4)  # Create a subplot in the fourth position.
plt.imshow(cropped, cmap="gray")  # Display the cropped ROI in grayscale.
plt.title("Cropped ROI")  # Set the title of the subplot.
plt.axis("off")  # Hide the axes.
plt.tight_layout()  # Adjust the layout for better visualization.

# Save and display the figure.
plt.savefig(
  caseImgPath.replace("Image.bmp", "Manipulation.jpg"),
  dpi=720,  # Set the resolution of the saved image.
  bbox_inches="tight",  # Set the bounding box to include the entire figure.
)  # Save the figure as an image.
plt.show()  # Display the figure.
plt.close()  # Close the figure.
plt.clf()  # Clear the current figure.
