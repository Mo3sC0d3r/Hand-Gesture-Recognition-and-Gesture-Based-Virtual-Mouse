# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:17:31 2020

@author: Mo3sC0d3r
"""

from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras.engine import training_generator

classifier = load_model('Trained_model_time.h5')



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
#print(class_labels)
    

from pynput.mouse import Button, Controller
import wx

#get mouse variable and screen size
mouse = Controller()
app = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(camx,camy) = (320,240) #setting the image resolution of captured image


cam = cv2.VideoCapture(0)

#previousmouse coord
mlocold=np.array([0,0])
dampingfactor=3
#after applyindamping
mouseloc=np.array([0,0])
flag=0


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
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    test_img = cv2.resize(mask, (image_x, image_y))
    
    
    # Finding contours 
    import imutils
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))
    cnts = imutils.grab_contours(cnts)
    
    test_image = image.img_to_array(test_img)
    x = test_image * 1./255
    test_image = np.expand_dims(x, axis = 0)
  
    result = classifier.predict_classes(test_image)

    result = class_labels[result[0]]
    #print(result)
    
    #  Loop over the contuors 
    for c in cnts:
        # Compute the center of the contour
        #if cv2.contourArea(c) > 0:
            #M = cv2.moments(c)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX= int(M["m10"] / M["m00"])
            cY= int(M["m01"] / M["m00"])
            
            # draw the contour and center of the shape on the image
            cv2.drawContours(imcrop, [c], -1, (0, 255, 0), 2)
            cv2.circle(imcrop, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(imcrop, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            
            if result == 'Move':
                if (flag == 1):
                    flag = 0
                    mouse.release(Button.left)
                
                #print("Testing cX: "+str((cX-40)*14)+" and cY: "+str((cY-20)*2))
                mouse.position = ((cX-70)*14, (cY-20)*2)
            if result == 'Left Click':
                mouse.press(Button.left)
                mouse.release(Button.left)
            
            if result == 'Right Click':
                mouse.press(Button.right)
                mouse.release(Button.right)


    cv2.putText(frame, result, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("test", frame)
    cv2.imshow("mask", mask) 
    
    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()
del app

