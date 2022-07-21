import urllib.request
import cv2
import numpy as np
import imutils
url = "http://192.168.43.112:8080/shot.jpg"    #connect mobile and laptop in same wifi network, preferable mobile hotspot
x =0                                          #install ip webcam app on mobile   
while x<2:                                    #open ip webcam app and swipe down to last and press start server          
    imgPath = urllib.request.urlopen(url)     #then copy the ipv4 address displayed on the mobile to the url part of the code  
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width = 450)
    #cv2.imshow("Snap", img)
    cv2.imwrite("Snap.jpg", img)
    x=x+1