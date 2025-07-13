from simple_image_download import simple_image_download

response = simple_image_download.simple_image_download()
response.download("batman joker", 68)

import os
from pygoogle_image import image as pi

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Download images
pi.download("corriandar", limit=25)
