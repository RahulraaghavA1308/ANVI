import imutils
import urllib.request
import cv2
import numpy as np

url = 'http://192.168.1.187:8080/shot.jpg'
flag = 0
while True:
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype = np.uint8)
        img = cv2.imdecode(imgNp, -1)
        img = imutils.resize(img, width = 450)
        cv2.imshow("camera", img)
        k =cv2.waitKey(1)
        # print("started")
        
        # print("done")q
        if ord('q') == k:
            cv2.imwrite("test.jpg", img)
            flag = 1
        elif ord('x') ==k:
            break
        
        if flag ==1:
            cv2.imshow("snap",cv2.imread("test.jpg"))