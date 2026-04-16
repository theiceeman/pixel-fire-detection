import cv2
import numpy as np

# read image
image = cv2.imread('fire.19.png')
# convert opencv BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for "Fire" colors (adjust these based on your specific photo)
lower_fire = np.array([0, 100, 200])   # Bright orange/yellow
upper_fire = np.array([30, 255, 255])

# Create a binary mask (white = fire, black = everything else)
mask = cv2.inRange(hsv, lower_fire, upper_fire)

# Clean up noise (remove tiny stray pixels)
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)