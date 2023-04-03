from flask import Flask, request, jsonify
import psycopg2
from PIL import Image
from flask_cors import CORS
import argparse
import os
import cv2
import pytesseract
import re
import datetime
import torch

cnls_cnfg = {
    'antelope_street': 'http://62.112.124.162:555/wBD02N2H?container=mjpeg&stream=main',
    # 'Zebra _street': 'http://62.112.124.162:555/QX0pbHKT?container=mjpeg&stream=main' ,
    'antelope_in_1': 'http://62.112.124.162:555/K0BRDzXf?container=mjpeg&stream=main',
    'antelope_in_2': 'http://62.112.124.162:555/k9Im7KGJ?container=mjpeg&stream=main',
    'antelope_street': 'http://62.112.124.162:555/KhPF0sKo?container=mjpeg&stream=main' ,
    'zhui_in_1': 'http://62.112.124.162:555/wBD02N2H?container=mjpeg&stream=main',
    'zhui_in_2': 'http://62.112.124.162:555/VHnz7z6J?container=mjpeg&stream=main',
    'dindin_in_1': 'http://62.112.124.162:555/GvuuEbAW?container=mjpeg&stream=main',
    'dindin_in_2': 'http://62.112.124.162:555/lnkqbFzK?container=mjpeg&stream=main',
    'dindin_street': 'http://62.112.124.162:555/chwnifID?container=mjpeg&stream=main',
    'zhui_street_1': 'http://62.112.124.162:555/F9gyCGH5?container=mjpeg&stream=main',
    'zhui_street_2': 'http://62.112.124.162:555/jJttvAGw?container=mjpeg&stream=main',
    'zhui_street_3': 'http://62.112.124.162:555/VluG8k6r?container=mjpeg&stream=main',
    'polyarnui_mir': 'http://62.112.124.162:555/w3VHg4cc?container=mjpeg&stream=main',
    'tokin_street': 'http://62.112.124.162:555/NU1zQyOW?container=mjpeg&stream=main',
    'tokin_in': 'http://62.112.124.162:555/u6gW8v2e?container=mjpeg&stream=main'
}

ANIMAL_CLASSES = ['cat', 'dog', 'horse', 'sheep', 'bird', 'bear', 'zebra', 'giraffe', 'bear']

dbname = 'animals'
user = 'postgres'
password = '1234'
host = 'localhost:5432'

torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = torch.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = torch.load('yolov5s.pt')
app = Flask(__name__)

def detection(cnls_cnfg, ANIMAL_CLASSES, model):
    res_dict = {}
    for cnl in cnls_cnfg:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        predicted_classes = []
        capture = cv2.VideoCapture(cnls_cnfg[cnl])
        ret, frame = capture.read()
        img = Image.fromarray(frame)
        results = model(img)

        # Access the predictions
        predictions = results.xyxy[0].cpu().numpy()

        # Get class names
        class_names = results.names
        predicted_classes = [class_names[int(prediction[5])] for prediction in predictions]


        # Check if there's an animal in the predicted classes
        is_animal_present = any([class_name in ANIMAL_CLASSES for class_name in predicted_classes])

        # Print 1 if there's an animal in the camera and 0 if there's no animal
        if is_animal_present:
            res_dict[cnl] = [1, dt_string]
        else:
            res_dict[cnl] = [0, dt_string]

    return res_dict


@app.route('/')
def hello():
    return 'Hello, Sasha'

@app.route('/animals',methods = [ 'GET'])
def realtime():
    results_ = detection(cnls_cnfg, ANIMAL_CLASSES, model)
    print(results_)
    return jsonify(results_)



if __name__ == '__main__':

    app.debug = True
    app.run()
