'''
#To install opencv write below command in anaconda prompt
conda install -c conda-forge opencv
'''
import cv2
#image
img=cv2.imread("C:\\Users\\kunjeshparekh\\Desktop\\Misc\\profile.jpg") #add image path in this with img name

cv2.imshow("Output",img)
cv2.waitKey(0) #0 is infinite delay , rest put the milli seconds delay

#video
cap=cv2.VideoCapture("C:\\Users\\kunjeshparekh\\Desktop\\Misc\\video_name.mp4") #add video path in this with video name

while True:
    success,img=cap.read() #same imgage(seq of img in video) in img 
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'): #this adds delay and and key press q to break loop
        break
    
#Webcam
cap=cv2.VideoCapture(0) #add webcam id if your computer has more than 1 web cams else 0 is default
cap.set(3,640) #width at ID=3
cap.set(4,480) #height at ID=4
cap.set(10,100) #brightness at ID=4

while True:
    success,img=cap.read() #same imgage(seq of img in video) in img 
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'): #this adds delay and and key press q to break loop
        break

