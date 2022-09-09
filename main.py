from logging import exception
import tensorflow as tf
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from tensorflow import keras
from keras import layers
from keras import activations

#(1) DATASET AND PREPROCESSING

dataset_path="E:\\gadhav\\python project\\dataset" #folder consisting of both masked and without masked images
groups=["masked2.0","unmasked2.0"] #list of folders inside dataset folder
for group in groups: #access each folder in dataset(ie masked2.0 and unmasked2.0)one by one
    path=os.path.join(dataset_path,group)
    for img in os.listdir(path):#access each img in the folder 
        img_array=cv.imread(os.path.join(path,img))
        plt.imshow(cv.cvtColor(img_array,cv.COLOR_BGR2RGB))#to convert BGR to RGB as color format in open CV is BGR not RGB
  
img_size=224#resize
new_array=cv.resize(img_array,(img_size,img_size))
plt.imshow(cv.cvtColor(new_array,cv.COLOR_BGR2RGB))



#(2) READ THE IMAGES AND CONVERTING IT INTO ARRAY

#convert images to array format
train_data=[]
def create_train_data():
    for group in groups:
        path=os.path.join(dataset_path,group)
        class_num=groups.index(group)               #labels
        for img in os.listdir(path):
            try:
                img_array=cv.imread(os.path.join(path,img))
                new_array=cv.resize(img_array,(img_size,img_size))
                train_data.append([new_array,class_num])
            except exception as e:
                pass
create_train_data() #each image saved as array appending one by one


random.shuffle(train_data)  #shuffle both set of emage arrays ie masked and unmasked
x=[] #data images
y=[] #labels masked or unmasked
for data,labels in train_data:
    x.append(data)
    y.append(labels)

x=np.array(x).reshape(-1,img_size,img_size,3)  #-1=all size of images,3=RGB ie channel
x=x/255.0 #normalization of data ie max level of gray level is 255
y=np.array(y)#covert lists to np array
pickle_out=open("x.pickle","wb")#pickle store data in form of byte stream
pickle.dump(x,pickle_out)
pickle_out.close()
pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
pickle_in=open("x.pickle","rb")
x=pickle.load(pickle_in)
pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)


#(3) DEEP LEARNING MODEL FOR TRAINING-TRANSFER LEARNING

model=tf.keras.applications.mobilenet.MobileNet()#pretrained image classifier

#transfer learning-tuning,weights starts from last check point(last 3 layers 
#and the last layer is represented as -1)
base_input=model.layers[0].input
base_output=model.layers[-4].output
#last 3 layers -3,-2,-1
final_output1=layers.Flatten()(base_output)
final_output2=layers.Dense(1)(final_output1)#0,1 ie masked or not
final_output3=layers.Activation(activations.sigmoid)(final_output2)
new_model=keras.Model(inputs=base_input,outputs=final_output2)

#finally setting it with or without face mask(binary classification 0/1)
new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
new_model.fit(x,y,epochs=1,validation_split=0.1)
new_model.save('my_model3.h5')
new_model=tf.keras.models.load_model('my_model3.h5')
from tkinter import Frame
from typing import final

path="E:\\gadhav\\python project\\env\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
font_scale=1.5

#set rectangular background to white
rectangle_bgr=(255,255,255)
font=cv.FONT_HERSHEY_SIMPLEX
#black image
img=np.zeros((500,500))
#set some text
text="some text in a box"
#get width and height of the box
(text_width,text_height)=cv.getTextSize(text, font, font_scale, thickness=1)[0]
#get text start position
text_offset_x=10
text_offset_y=img.shape[0]-25
box_coords=((text_offset_x,text_offset_y),(text_offset_x+text_width+2,text_offset_y-text_height-2))
cv.rectangle(img,box_coords[0],box_coords[1],rectangle_bgr,cv.FILLED)
cv.putText(img,text,(text_offset_x,text_offset_y),font,fontScale=font_scale,color=(0,0,0),thickness=1)

cap=cv.VideoCapture(0)
#check if thge webcam is opened coreectly
if not cap.isOpened():
    cap=cv.VideoCapture(1)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret,frame=cap.read()
    faceCascade=cv.CascadeClassifier(path)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        faces=faceCascade.detectMultiScale(roi_gray)
        if len(faces)==0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in faces:
                face_roi=roi_color[ey:ey+eh, ex:ex+ew]#cropping the face
    
    final_image=cv.resize(face_roi,(244,244))
    final_image=np.expand_dims(final_image,axis=0)#need fourth dimension
    final_image=final_image/255.0
    font=cv.FONT_HERSHEY_SIMPLEX
    predictions=new_model.predict(final_image)
    font_scale=1.5
    font=cv.FONT_HERSHEY_PLAIN

    if (predictions>0):
        status="NO Mask"
        x1,y1,w1,h1=0,0,175,75
        #draw black background rectangle
        cv.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        #add text
        cv.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv.putText(frame,status(100,150),font,3,(0,0,255),2,cv.LINE_4)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

    else:
        status="Face Mask"
        x1,y1,w1,h1=0,0,175,75
           #draw black background rectangle
        cv.rectangle(frame,(x1,x1),(x1+w1,y1+h1),(0,0,0),-1)
        #add text
        cv.putText(frame,status,(x1+int(w1/10),y1+int(h1/2)),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv.putText(frame,status(100,150),font,3,(0,255,0),2,cv.LINE_4)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))

        cv.imshow('FACE MASK DETECTION',frame)

        if cv.waitKey(2) & 0xFF==ord('q'):
            break
cap.release()
cv.destroyAllWindows()