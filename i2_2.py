import cv2
img = cv2.imread("a.png")
go= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ti = cv2.threshold(img,125,255,cv2.THRESH_BINARY)[1]
gaussianBlurImg = cv2.GaussianBlur(img,(21,21),0)
cv2.imwrite("gb.jpg",gaussianBlurImg)
cv2.imshow("gb.jpg",gaussianBlurImg)
cv2.imshow("sxh",ti)
