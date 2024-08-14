# -*- coding: utf-8 -*-
"""
Created on ago 2024

@author: Alfonso Blanco
"""
#######################################################################
# PARAMETERS
######################################################################
# dataset
# https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets/data?select=sagittal_t1wce_2_class


dirname= "Test1\\images"
dirnameLabels="Test1\\labels"



#dirnameYolo="runs\\train\\exp2\\weights\\best.pt"

#dirnameYolo="last14epoch9hits.pt"
#dirnameYolo="last27epoch7x0.pt"  # la primera
#dirnameYolo="last41epoch4x8.pt"
#dirnameYolo="lastepoch24-4x4.pt" # muy mala
#dirnameYolo="best.pt"

dirnameYolo1="last27epoch7x0.pt"
dirnameYolo2="last14epoch9hits.pt"

import cv2
import time
Ini=time.time()

#from ultralytics import YOLOv10
from ultralytics import YOLO

#model = YOLOv10(dirnameYolo)
model1 = YOLO(dirnameYolo1)
model2 = YOLO(dirnameYolo2)

class_list = model1.model.names
print(class_list)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import imutils

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 
                 image = cv2.imread(filepath)
                 #print(filepath)
                 #print(image.shape)                           
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
     
     Labels = []
     TabFileLabelsName=[]
     Tabxyxy=[]
     ContLabels=0
     ContNoLabels=0
         
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 Label=""
                 xyxy=""
                 for linea in f:
                      #print(filename)
                      #print(linea)
                      indexFracture=int(linea[0])
                      
                      #if indexFracture==0:
                      #     indexFracture=1
                      #else:
                      #      indexFracture=0
                            
                      Label=class_list[indexFracture]
                      #print(Label)
                      xyxy=linea[2:]
                      
                                            
                 Labels.append(Label)
                 
                 if Label=="":
                      ContLabels+=1
                 else:
                     ContNoLabels+=1 
                 
                 TabFileLabelsName.append(filename)
                 Tabxyxy.append(xyxy)
     return Labels, TabFileLabelsName, Tabxyxy, ContLabels, ContNoLabels

def unconvert(width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)

    return xmin, ymin, xmax, ymax

# ttps://medium.chom/@chanon.krittapholchai/build-object-detection-gui-with-yolov8-and-pysimplegui-76d5f5464d6c
def Detect_sagittal_t1wceWithYolov10 (img):
  
   Tabcrop_sagittal_t1wce=[]
   
   y=[]
   yMax=[]
   x=[]
   xMax=[]
   Tabclass_name=[]
   Tabclass_cod=[]
   Tabconfidence=[]

   cont=0
   model=model2 # por defecto model2
   while cont < 3:
        SwHay=0
        cont=cont+1
        if cont==1:
           results = model1(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                #print("FALLA1")
                continue
               SwHay=1
               model=model1
               break
          
        if SwHay==1 : break 
        if cont==2:
          
           
           results = model2(source=img)
           
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                        continue
               SwHay=1
               model=model2
               break
        if SwHay==1 : break
        """
        if cont==3:
           
           results = model3(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                      continue
               SwHay=1
               model=model3
               break

        if cont==4:
           results = model4(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                #print("FALLA1")
                continue
               SwHay=1
               model=model4
               break
          
        if SwHay==1 : break
        
        if cont==5:
           results = model5(source=img)
           for j in range(len(results)):
               # may be several plates in a frame
               result=results[j]
       
               xyxy= result.boxes.xyxy.numpy()
               confidence= result.boxes.conf.numpy()
               class_id= result.boxes.cls.numpy().astype(int)
               #print("Class_id" )
               #print(class_id)
               #print("results...."+ str(len(results)))
               if len(class_id)==0 :
                #print("FALLA1")
                continue
               SwHay=1
               model=model5
               break

        
        if SwHay==1 : break
        """
        continue
   
   
   # https://blog.roboflow.com/yolov10-how-to-train/
   results = model(source=img)
   for i in range(len(results)):
       # may be several plates in a frameh
       result=results[i]
       
       xyxy= result.boxes.xyxy.numpy()
       confidence= result.boxes.conf.numpy()
       class_id= result.boxes.cls.numpy().astype(int)
       print(class_id)
       # diference .txt labels y data.yaml
       #for h in range(len(class_id)):
       #     if class_id[h]==0:
       #          class_id[h]=1
       #     else:
       #          class_id[h]=0
       out_image = img.copy()
       LabelTotal=""
       for j in range(len(class_id)):
           con=confidence[j]
           Tabconfidence.append(con)
           # due to arror assignement of classes in yaml
           #if class_id[j]==0 :
           #    class_id[j]=1
           #else:
           #    class_id[j]=0 
           label=class_list[class_id[j]] + " " + str(con)[0:4]
           print(label)
           LabelTotal=LabelTotal+" " + label
           box=xyxy[j]
           
           crop_sagittal_t1wce=out_image[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
           
           Tabcrop_sagittal_t1wce.append(crop_sagittal_t1wce)
           y.append(int(box[1]))
           yMax.append(int(box[3]))
           x.append(int(box[0]))
           xMax.append(int(box[2]))

           # 
           Tabclass_name.append(label)
           Tabclass_cod.append(class_id[j])
          
   
   return Tabconfidence, Tabcrop_sagittal_t1wce, y,yMax,x,xMax, Tabclass_name, Tabclass_cod, LabelTotal

def plot_image(image, boxes, boxesTrue, imageCV, TabFileName):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    #class_labels = PASCAL_CLASSES
    class_labels=class_list
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    fig.suptitle(TabFileName)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    Cont=0
    print(boxes)
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        conf=box[1]
        conf=str(conf)
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)] + " conf: " + str(conf[:3]),
            color="red",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )
      
        

        
        Cont+=1
        #if Cont > 1: break # only the most predicted box
        #break
      # rect with true fracture
   
    plt.show()

###########################################################
# MAIN
##########################################################

Labels, TabFileLabelsName, TabxyxyTrue, ContLabels, ContNoLabels= loadlabels(dirnameLabels)

print("Number of images to test : " + str(len(Labels)))

print("Number of files without labels : " + str(ContNoLabels))
print("Number of files with labels : " + str(ContLabels))


imagesComplete, TabFileName=loadimages(dirname)

print("Number of images to test: " + str(len(imagesComplete)))

ContError=0
ContHit=0
ContNoDetected=0

for i in range (len(imagesComplete)):
            
            if TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] != TabFileName[i][:len(TabFileName[i])-4]:
                 print("ERROR SEQUENCING IMAGES AN LABELS " + TabFileLabelsName[i][:len(TabFileLabelsName[i])-4] +" --" + TabFileName[i][:len(TabFileName[i])-4])
                 break
            # no se consideran las que no vienen labeladas
            if Labels[i] == "": continue
            gray=imagesComplete[i]
           
            imgTrue=imagesComplete[i]
           
            XcenterYcenterWH=TabxyxyTrue[i].split(" ")
            width=float(imgTrue.shape[0])
            height=float(imgTrue.shape[1])
            x=float(XcenterYcenterWH[0])
            y=float(XcenterYcenterWH[1])
            w=float(XcenterYcenterWH[2])
            h=float(XcenterYcenterWH[3])
            xTrue,yTrue,xMaxTrue,yMaxTrue=unconvert(width, height, x, y, w, h)
           
            start_pointTrue=(int(xTrue),int(yTrue)) 
            end_pointTrue=(int(xMaxTrue),int( yMaxTrue))
           
            colorTrue=(0,0,255)
            
            # Using cv2.rectangle() method
            # Draw a rectangle with green line borders of thickness of 2 px
            imgTrue = cv2.rectangle(imgTrue, start_pointTrue, end_pointTrue,(0,255,0), 2)
           
            Tabconfidence, TabImgSelect, y, yMax, x, xMax, Tabclass_name, Tabclass_cod, LabelTotal =Detect_sagittal_t1wceWithYolov10(gray)
            Tabnms_boxes=[]
            #print(gray.shape)
            if TabImgSelect==[]:
                print(TabFileName[i] + " NON DETECTED")
                ContNoDetected=ContNoDetected+1 
                continue
            else:
                #ContDetected=ContDetected+1
                print(TabFileName[i] + " DETECTED ")
                
               
            for z in range(len(TabImgSelect)-1,0, -1):
                #if TabImgSelect[z] == []: continue
                gray1=TabImgSelect[z]
                #cv2.waitKey(0)
                # may be several tumors, positives and negatives
                #print(x[z])
                text_color = (255,255,255)
                
                cv2.putText(gray, LabelTotal ,(20,20)
                             , cv2.FONT_HERSHEY_SIMPLEX , 1
                             , text_color, 2 ,cv2.LINE_AA)
                
                start_point=(x[z],y[z]) 
                end_point=(xMax[z], yMax[z])
                #print("Tabclass_cod[z] = " + str(Tabclass_cod[z]))
                if Tabclass_cod[z] == 0:
                        #positive red, negative blue
                        color=(255,0,0)
                else:
                        color=(0,0,255)  
                     # Using cv2.rectangle() method
                     # Draw a rectangle with blue line borders of thickness of 2 px
                #print("crea rectangulo")
                img = cv2.rectangle(gray, start_point, end_point,color, 2)
            print("Is labeled as " + Labels[i])    
            plot_image(img, Tabnms_boxes, TabxyxyTrue[i], img, TabFileName[i])
                
             
              
print("")           
print("NO detected=" + str(ContNoDetected))
#print("Errors positive negative =" + str(ContError))

print("")      
print( " Time in seconds "+ str(time.time()-Ini))
