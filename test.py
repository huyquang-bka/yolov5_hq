import torch
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'data/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
t = time.time()
results = model(img)
print("Time: ", time.time() - t)
# Results
results.show()
