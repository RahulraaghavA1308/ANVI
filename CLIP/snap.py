import urllib.request
import cv2
import numpy as np
import imutils

url = 'http://192.168.1.105:8080/shot.jpg'

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype = np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width = 450)
    cv2.imshow("camera", img)
    k =cv2.waitKey(1)
    # print("started")
    cv2.imwrite("try.jpg", img)
    # print("done")
    if ord('q') == k:
        break