import cv2
#image
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\Misc\\profile.jpg") #add image path in this with img name

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #for gray scale img from color
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1) #(7,7) has to be kernal size must be (odd,odd) for blurring img and blurr 1 is higer the sigma higher the blurr
imgCanny=cv2.Canny(img,100,100) #edge detection for lesser edge use higher number like 200,150
import numpy as np
kernal=np.ones((3,3),np.uint8)
imgDialation=cv2.dilate(imgCanny,kernal,iterations=1) #increase the thickness of edge henece edge img imgCanny is supplied
#higher the iteration higher the thickness of edges
imgEroded=cv2.erode(imgDialation,kernal,iterations=1) #decrease the thickness of edge
#higher the iteration lower the thickness of edges

cv2.imshow("image",img)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay


cv2.imshow("Gray image",imgGray)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("Blur image",imgBlur)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("Canny image",imgCanny)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("Dialation image",imgDialation)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("Eroded image",imgEroded)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay


#Resize the img

import cv2
#image
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\Misc\\profile.jpg") #add image path in this with img name
print(img.shape) #check the img size (height,width,color channel)
imgresize=cv2.resize(img,(400,300)) #here 400 is width and then 450 is height
imgCropped=img[75:200,100:250] #height first width later


cv2.imshow("image",img)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("image Resize",imgresize)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

cv2.imshow("Cropped image",imgCropped)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

#How to draw shapes in img and add texts 
#0 is black and 1 is white in every text

import cv2
import numpy as np
img=np.zeros((512,512))
img=np.zeros((512,512,3),np.uint8) #To add color channel added 3
img[:]=255,0,0 #now for blue img
img[200:300,100:300]=255,0,0 #HXWXdim delet above row if you want a small blue box 
cv2.line(img,(0,0),(300,300),(0,255,0),3) #Line from 00 to 300,300 coordinates with line color = 0,255,0 and optional thickness of 3
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2) #rectangle from 00 to 300,300 coordinates 
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED) #to fill the rectangle with 0,0,255 i.e. red color
cv2.circle(img,(400,50),30,(255,255,0),5) #center 400,50 radius 30 and 5 thickness and color 255,255,0
cv2.putText(img,"OpenCV ",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),3) #start text from 300,100, 1 is size of text ,3 thickness

cv2.imshow("image_black",img)
cv2.waitKey(0)


#WARP perspective to get the bird's eye view from slant img to straight image like the cards one shown below
import cv2
import numpy as np
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\cards.png") #add image path in this with img name
cv2.imshow("cardimg",img)
cv2.waitKey(0)

img=cv2.resize(img,(500,500))
cv2.imwrite("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\new.png",img)

width,height=250,350
pts1=np.float32([[111,229],[287,200],[154,482],[385,430]]) #rop left,top right,bottom left,bottom right points in paint
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]]) #resultant img's height width
matrix=cv2.getPerspectiveTransform(pts1,pts2)
imgoutput=cv2.warpPerspective(img,matrix,(width,height))
cv2.imshow("op",imgoutput)
cv2.waitKey(0)

#Joining Images in horizontal & Vertical Stack
import cv2
import numpy as np
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\cards.png") #add image path in this with img name
horizontal_stk=np.hstack((img,img))
cv2.imshow("horizontal",horizontal_stk)
cv2.waitKey(0)

vertical_stk=np.vstack((img,img,img))
cv2.imshow("Vertical",vertical_stk)
cv2.waitKey(0)

#Horizontal+Vertical Stack
imghv=np.vstack(((np.hstack((img,img,img))),np.hstack(((img,img,img))),np.hstack((img,img,img))))

cv2.imshow("HVStack",imghv)
cv2.waitKey(0)

#Color Detection
'''
HSV:Hue, Saturation, and Value (HSV) is a color model that is often used in place of the RGB color model in graphics 
and paint programs. In using this color model, a color is specified then white or black is added to easily make 
color adjustments. HSV may also be called HSB (short for hue, saturation and brightness).
'''
import cv2
import numpy as np
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\car.png") #add image path in this with img name
cv2.imshow("image",img)
cv2.waitKey(0)

imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",imgHSV)
cv2.waitKey(0)

hstk=np.hstack((img,imgHSV))
cv2.imshow("HVStack",hstk)
cv2.waitKey(0)
    #Add tracker window
def empty(a):
    pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240) #Trackbar's size
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty) #Hue can have max value from 0 to 179 not above that
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\car.png") #add image path in this with img name
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min=cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max=cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min=cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max=cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min=cv2.getTrackbarPos("Val Min","TrackBars")
    v_max=cv2.getTrackbarPos("Val Max","TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Image",img)
    cv2.imshow("Mask",mask)
    cv2.waitKey(1)
#For setting this up we need to keep all the colors which we want in white and rest in Black
#0,179,93,255,112,255  best fit with all the required colors in white and rest in Black
#Same above code but now with optimal parameters as above
def empty(a):
    pass
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240) #Trackbar's size
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty) #Hue can have max value from 0 to 179 not above that
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",93,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",112,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\car.png") #add image path in this with img name
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min=cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max=cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min=cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max=cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min=cv2.getTrackbarPos("Val Min","TrackBars")
    v_max=cv2.getTrackbarPos("Val Max","TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(imgHSV,lower,upper)
    imgResult=cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("HSV",imgHSV)
    cv2.imshow("Image",img)
    cv2.imshow("Mask",mask)
    cv2.imshow("result",imgResult)
    cv2.waitKey(1) #ALLOWS wait for a sec and loops through while

#Contours/Shape Detection
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\KP\\Opencv\\shapes.png") 
cv2.imshow("Image",img)
cv2.waitKey(0) 

imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #for gray scale img from color
cv2.imshow("ImageGray",imgGray)
cv2.waitKey(0) 

imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
cv2.imshow("ImageBlur",imgBlur)
cv2.waitKey(0) 

imgCanny=cv2.Canny(img,50,50)
cv2.imshow("ImageCanny",imgCanny)
cv2.waitKey(0) 

cv2.imshow("ImageStack",np.hstack((imgGray,imgCanny,imgBlur)))
cv2.waitKey(0) 
imgContour=img.copy()
def getContours(img):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL only find extreme extrenal cotours (Good With Outer countours) there are other methods to find internal contours & later the argument chain approx none is calling all the contours
    for cnt in contours:
        area=cv2.contourArea(cnt)
        print(area)
        #cv2.drawContours(imgContour,cnt,-1,(255,0,0),2)
        if area>50:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),2) #as we can see in the area that for few contours the area is blank 255,0,0 => BGR shows blue color contour with thickness 2
            peri=cv2.arcLength(cnt,True)
            print(peri)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True) #Gives corner points of each shapes
            print("KP",approx,len(approx)) #Any length abive 4 can be assumsed as circle 
            objCor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            if objCor==3:
                ObjectType="Tri"
            elif objCor==4 and w/h>0.95 and w/h < 1.05:
                ObjectType="Sq"
            elif objCor==4:
                ObjectType="Rect"
            elif objCor>4:
                ObjectType="Cir"
            else:
                ObjectType="XX"
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2) #Bounding Box across each object  
            cv2.putText(imgContour,ObjectType,((x+w//2)-10,(y+h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1) #x+w//2 is the area for where to write text
            
getContours(imgCanny) #Gives area of imgcanny

cv2.imshow("ImageContour",imgContour)
cv2.waitKey(0) #Shows blue line of enclosed surfaces

#Face Detection

'''
We will need haarcascade files by opencv which are pretrained models for face detection + there are 
others like eye,numberpalte etc pre-trained models available as well. You name the object and the 
pretrained models are available.
'''

import cv2
faceCascade=cv2.CascadeClassifier("C:\\Users\\kunjeshparekh\Desktop\\KP\\Opencv\\haarcasade_frontal\\haarcascade_frontalface_default.xml")
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\Misc\\profile.jpg") #for 1 face comment below line
img=cv2.imread("C:\\Users\\kunjeshparekh\Desktop\\KP\\Opencv\\multiple_faces.jpg") #For multiple faces
img=cv2.imread("C:\\Users\\kunjeshparekh\Desktop\\KP\\Opencv\\mask_img.jpg") #For multiple mask faces
img=cv2.imread("C:\\Users\\kunjeshparekh\Desktop\\KP\\Opencv\\mask_img1.jpg") #For 1mask face

cv2.imshow("image",img)
cv2.waitKey(0) 
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
faces=faceCascade.detectMultiScale(imgGray,1.1,4) #Scale factor = 1.1,min neighbours=4 can change based on results

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #Create bounding box around the img
    cv2.rectangle(imgGray,(x,y),(x+w,y+h),(0,0,0),4) #Create bounding box around the img
    print(x,y,w,y)
cv2.imshow("Result",img)
cv2.waitKey(0) 

cv2.imshow("Result",imgGray)
cv2.waitKey(0) 
