import imutils
import cv2

redLower = (38, 62, 100)
redUpper = (179, 255, 255)

camera=cv2.VideoCapture(0)

while True:

        (grabbed, frame) = camera.read()

        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)


        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 10:
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        print(center,radius)
                        if radius > 250:
                                print("stop")
                        else:
                                if(center[0]<150):
                                    pos = "Left"
                                elif(center[0]>450):
                                    pos = "Right"
                                elif(radius<250):
                                    pos = "Front"
                                else:
                                    pos = "Stop"
                                print(pos)
                        cv2.putText(frame, pos, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break

camera.release()
cv2.destroyAllWindows()
