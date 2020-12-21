# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:44:47 2020

@author: Mo3sC0d3r
"""


from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.engine import training_generator

classifier = load_model('Trained_model_present.h5')



num_classes = 7
img_rows, img_cols = 64, 64
batch_size = 16

def nothing(x):
    pass

image_x, image_y = 64,64

train_data_dir = 'mydata\\training_set'
validation_data_dir = 'mydata\\test_set'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      shear_range=0.3,
      zoom_range=0.3,
      width_shift_range=0.4,
      height_shift_range=0.4,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)
    

cam = cv2.VideoCapture(0)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("test")

img_counter = 0

img_text = ''
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")


    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    imcrop = img[102:298, 427:623]
    #imcrop = img[100:300,425:625]
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
      
    test_image = cv2.resize(mask, (image_x, image_y))
    
    test_image = image.img_to_array(test_image)
    x = test_image * 1./255
    test_image = np.expand_dims(x, axis = 0)
    #test_image = np.vstack([test_image])
    #print(test_image.shape)
    #result = classifier.predict(test_image)
    result = classifier.predict_classes(test_image)
    #print(result[0])
    result = class_labels[result[0]]
    #print(result)
    
    
    cv2.putText(frame, result, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask) 
    
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()