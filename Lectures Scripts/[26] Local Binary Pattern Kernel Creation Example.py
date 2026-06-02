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
from HMB_Spring_2026_Helpersac import *

# Set the distance parameter for the Local Binary Pattern (LBP) computation.
distance = 1

# Specify whether the LBP computation should follow a clockwise direction.
isClockwise = True

# Define the starting angle (theta) for the LBP computation, measured in degrees.
thetaDegree = 135  # Start from the top-left corner.

# Create a kernel matrix to represent the LBP pattern weights.
kernel = BuildLBPKernel(
  distance=distance,  # Distance from the center pixel to the surrounding pixels.
  theta=thetaDegree,  # Angle in degrees for the kernel rotation.
  isClockwise=isClockwise,  # Direction of rotation (True for clockwise, False for counterclockwise).
)
print(f"Kernel for distance {distance}, theta {thetaDegree}, and clockwise {isClockwise}:")
print(kernel)  # Print the generated kernel matrix.
