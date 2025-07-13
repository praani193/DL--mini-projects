#install ip webcam app in moble and get url
import urllib.request
import cv2
import numpy as np
import imutils
url='http://172.19.111.77:8080/shot.jpg'
while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array (bytearray (imgPath.read()), dtype=np.uint8)
    img= cv2.imdecode (imgNp, -1)
    img= imutils.resize(img, width=450)
    cv2.imshow("CameraFeed", img)
    if ord('g')== cv2.waitKey(1):
        exit(0)
