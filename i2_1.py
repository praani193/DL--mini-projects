import cv2
import imutils
img = cv2.imread("a.png")
r =imutils.resize(img,width=550)
cv2.imwrite("rimg.jpg",r)
cv2.imshow("a",img)
cv2.imshow("a1",r)
