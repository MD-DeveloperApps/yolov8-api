from fastapi import FastAPI
from PIL import Image, ImageDraw
from flask import request, Flask, jsonify, send_file
from ultralytics import YOLO
from waitress import serve
from io import BytesIO
import base64
import json
import datetime
import random
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = 'models'

MODELNAME = app.config['MODELS_FOLDER']
NAME = 'yolov8n.pt'

@app.route("/")
def root():
    with open('index.html') as f:
        return f.read()


@app.route("/detect", methods=['POST'])
def detect():
    buf= request.files['image_file']
    show= request.form['show'] 
    model= request.form['model']
    existmodel = os.path.isfile(MODELNAME+'/'+model)
    if(existmodel!=True):
        return jsonify({'status': 'error', 'message': 'Model not found'})
    NAME = model
    if(show=='BLOB'):
        boxes = detect_objects_image(buf.stream)
        return jsonify(boxes)
    elif(show=='URL'):
        url = detectandsave(buf.stream)
        return jsonify(url)
    elif(show=='PARAMETERS'):
        boxes = detect_objects_on_image(buf.stream)
        return jsonify(boxes)
    return jsonify({'status': 'error', 'message': 'Invalid request'})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    print(filename)
    return send_file(f'uploads/{filename}', mimetype='image/jpeg')


#utils

def getDateStr():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def detectandsave(buf):
    model = YOLO(MODELNAME+'/'+NAME)
    results = model.predict(Image.open(buf))
    result = results[0]
    img= blobWithBoxes(buf, result.boxes)
    filename = secure_filename(getDateStr()+random.randint(1, 1000000).__str__()+'.jpg')
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    url = f'{request.host_url}{UPLOAD_FOLDER}/{filename}'
    return url
    
def detect_objects_on_image(buf):
    model = YOLO(MODELNAME+'/'+NAME)
    results = model.predict(Image.open(buf))
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output


def detect_objects_image(buf):
    model = YOLO(MODELNAME+'/'+NAME)
    results = model.predict(Image.open(buf))
    result = results[0]
    filename = secure_filename(getDateStr()+random.randint(1, 1000000).__str__()+'.jpg')
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return image_to_json(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def image_to_json(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    encoded_image = 'data:image/jpeg;base64,' + encoded_image
    json_data = {
        'image': encoded_image,
        'name': 'output.jpg'
    }
    json_string = json.dumps(json_data, separators=(',', ':'),indent=2, sort_keys=True).replace('\\x','')

    return json_string

def blobWithBoxes(image, boxes):
    """
    Function receives an image and an array of bounding boxes
    and returns an image with bounding boxes drawn on it
    :param image: Input image file stream
    :param boxes: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    :return: Output image file stream
    """
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box.xyxy[0].tolist(), outline='red')
        draw.text(
            box.xyxy[0, :2].tolist(),
            f'{box.cls[0]} {box.conf[0]:.2f}',
            fill='red'
        )

    return img

print('Server is running on port 8000')   
serve(app, host='0.0.0.0', port=8000)
