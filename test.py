import cv2
import numpy as np

# Create a blank image with different data types
width, height = 800, 800

# 8-bit unsigned integer (uint8) image
uint8_image = np.zeros((height, width), dtype=np.uint8)

# 16-bit unsigned integer (uint16) image
uint16_image = np.zeros((height, width), dtype=np.uint16)

# 32-bit floating-point (float32) image
float32_image = np.zeros((height, width), dtype=np.float32)

# 64-bit floating-point (float64) image
float64_image = np.zeros((height, width), dtype=np.float64)

# Fill each image with different colors
uint8_image[50:150, 50:150] = 255
uint16_image[50:150, 150:250] = 65535
float32_image[150:250, 50:150] = 1.0
float64_image[150:250, 150:250] = 1.0

# Concatenate the images horizontally for display
combined_image = np.hstack((uint8_image, uint16_image, float32_image, float64_image))

# Display the combined image
cv2.imshow('Different Data Types', combined_image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()


