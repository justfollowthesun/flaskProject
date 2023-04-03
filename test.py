from config import *
import cv2
from PIL import Image
import torch


torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
ANIMAL_CLASSES = ['cat', 'dog', 'horse', 'sheep', 'bird', 'bear', 'zebra', 'giraffe', 'bear']
for cnl in cnls_cnfg:
  predicted_classes = []
  capture = cv2.VideoCapture(cnls_cnfg[cnl])
  ret, frame = capture.read()
  img = Image.fromarray(frame)
  results = model(image)

  # Access the predictions
  predictions = results.xyxy[0].cpu().numpy()

  # Get class names
  class_names = results.names
  predicted_classes = [class_names[int(prediction[5])] for prediction in predictions]

  print(predicted_classes)

  # Check if there's an animal in the predicted classes
  is_animal_present = any([class_name in ANIMAL_CLASSES for class_name in predicted_classes])

  # Print 1 if there's an animal in the camera and 0 if there's no animal
  if is_animal_present:
      print(1)
  else:
      print(0)
