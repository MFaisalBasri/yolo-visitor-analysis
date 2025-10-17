# import cv2 as cv
# import torch

# build = cv.getBuildInformation()
# if 'CUDA' in build:
#     print("OpenCV is built with CUDA support.")
# if torch.cuda.is_available():
#     print("Torch is using CUDA.")

# import torch
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())
# print(torch.version.cuda)

import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


