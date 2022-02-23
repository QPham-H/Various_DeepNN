import numpy as np
import pandas as pd
import tensorflow as tf

import cv2
import tensorflow_hub
import matplotlib.pyplot as plt

# Constants
img_wd = 256
img_ht = 256
file = 'sample.jpg'
score_threshold = 0.3


# Load image
img = cv2.imread(file)
img = cv2.resize(img, (img_wd,img_ht))

# If img is BGR, convert to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# The input tensor is a tf.uint8 tensor with shape [None, height, width, 3] with values in [0, 255]
img_tensor = tf.convert_to_tensor(img) #,dtype=tf.uint8)
img_tensor = tf.expand_dims(img_tensor, axis = 0)

# Load object detection model using EfficientDet Lite from Tensorflow Hub
model_link = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1"
detector = hub.load(model_link)

# Creating prediction
boxes, scores, classes, num_detections = detector(rgb_tensor)

##labels = pd.read_csv('labels.csv', sep=';', index_col='ID')
##labels = labels['OBJECT (2017 REL.)']
##
##pred_labels = classes.numpy().astype('int')[0] 
##pred_labels = [labels[i] for i in pred_labels]
##pred_boxes = boxes.numpy()[0].astype('int')
##pred_scores = scores.numpy()[0]


for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
    if score < score_threshold:
        continue

    score_txt = f'{100 * round(score)}%'
    img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),2)      
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_boxes, label,(xmin, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 1.5, (255,0,0), 2, cv2.LINE_AA)

# Live Webcam Video
import tensorflow_hub as hub
import cv2
import numpy
import tensorflow as tf
import pandas as pd

# Carregar modelos
detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
labels = labels['OBJECT (2017 REL.)']

cap = cv2.VideoCapture(0)

width = 512
height = 512

while(True):
    #Capture frame-by-frame
    ret, frame = cap.read()
    
    #Resize to respect the input_shape
    inp = cv2.resize(frame, (width , height ))

    #Convert img to RGB
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    boxes, scores, classes, num_detections = detector(rgb_tensor)
    
    pred_labels = classes.numpy().astype('int')[0]
    
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]
   
   #loop throughout the detections and place a box around it  
    for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue
            
        score_txt = f'{100 * round(score,0)}'
        img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes,label,(xmin, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(img_boxes,score_txt,(xmax, ymax-10), font, 0.5, (255,0,0), 1, cv2.LINE_AA)



    #Display the resulting frame
    cv2.imshow('black and white',img_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
