# from super_image import EdsrModel, ImageLoader
# from PIL import Image
# import requests
# import cv2

# url = '/home/pratyush/Downloads/download.jpeg'
# # image.show()
# image = Image.open(url)


# model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)      
# inputs = ImageLoader.load_image(image)
# preds = model(inputs)

# ImageLoader.save_image(preds, './scaled_2x.png')                       
# ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')     


import cv2
import numpy as np
from super_image import EdsrModel, ImageLoader
from PIL import Image

url = '/home/pratyush/Downloads/download.jpeg'
image = Image.open(url)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

# Convert PIL images to NumPy arrays for display using OpenCV
input_np = np.array(inputs).astype(np.uint8)  # Ensure data type is uint8

# Convert the list of PyTorch tensors to a single NumPy array
preds_np = np.array([pred.detach().numpy() for pred in preds])

# Additional parameters for saving as JPEG with optimization
params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]

# Save the super-resolved image with additional parameters
cv2.imwrite('output_image.jpg', preds_np, params)

