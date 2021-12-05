# -*- coding: utf-8 -*-
"""

@author: jai ganesh
"""


import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

#load model
model = load_model("gender_predictor.h5")

from mtcnn.mtcnn import MTCNN

#creating object
detector=MTCNN()



#definig classes
classes = ['men','women']


def gender_facecounter(image,m,f,size=0.5):
    
    faces=detector.detect_faces(image)
    # iterating over faces 

    for face in range(len(faces)):
        
        boundingbox=faces[face]['box']
        x=boundingbox[0]
        y=boundingbox[1]
        w=boundingbox[2]
        h=boundingbox[3]

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),3)
    
        cropped_image = np.copy(image[y:y+h,x:x+w])
    
        #preprocess the image according to our model
        res_face = cv2.resize(cropped_image, (96,96))
        #cv2.imshow("cropped image",res_face)
        res_face = res_face.astype("float") / 255.0
        res_face = img_to_array(res_face)
        res_face = np.expand_dims(res_face, axis=0)
        
    
        #model prediction
        result = model.predict(res_face)[0] 
    
        # get label with max accuracy
        idx = np.argmax(result)
        label = classes[idx]
    
        #calculating count
        if label=="women":
            f=f+1
        else:
            m=m+1

    cv2.rectangle(image,(0,0),(300,30),(255,255,255),-1)
    cv2.putText(image," females = {},males = {} ".format(f,m),(0,15),cv2.FONT_HERSHEY_TRIPLEX,0.6,(255, 101, 125),1)
    cv2.putText(image," faces detected = " + str(len(faces)),(10,30),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,0),1)
    
    return image
    
    

source = cv2.VideoCapture(0)

while True:
    ret, frame = source.read()
    x=0
    y=0
    cv2.imshow("Live Facecount",gender_facecounter(frame,x,y))
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
           
source.release() 
cv2.destroyAllWindows()
