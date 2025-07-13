import cv2
import time
import imutils

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None
area = 500
object_count = 0

while True:
    _, img = cam.read()
    text = "Normal"
    img = imutils.resize(img, width=500)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussimg = cv2.GaussianBlur(gimg, (21, 21), 0)
    
    if firstFrame is None:
        firstFrame = gaussimg
        continue
    
    imgDiff = cv2.absdiff(firstFrame, gaussimg)
    t = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
    t = cv2.dilate(t, None, iterations=2)
    cnts = cv2.findContours(t.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
        object_count += 1  
        print(text)
    
    
    cv2.putText(img, f"Objects Count: {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print("Object Count:", object_count)
    
    cv2.imshow("imgfromcam", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
