import cv2
import time
time.sleep(1)
cam = cv2.VideoCapture(0)
while True :
    _,img =cam.read()
    cv2.imshow("imgfromcam",img)
    key = cv2.waitKey(1)&0xFF
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()


