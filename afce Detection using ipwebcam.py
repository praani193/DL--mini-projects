import cv2
import numpy as np
import os
import urllib.request
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

datasets = 'dataset'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print("Subject Path:", subjectpath)  # Debugging
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            print("Image Path:", path)  # Debugging
            label = id
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Error loading image:", path)
                continue
            print("Image shape:", image.shape)  # Debugging
            images.append(image)
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]
print("Number of images loaded:", len(images))  # Debugging
print("Labels:", labels)  # Debugging

(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)
url = "http://192.168.137.158:8080/shot.jpg"
cnt = 0

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        faceCrop = grayImg[y:y+h, x:x+w]
        resizedFace = cv2.resize(faceCrop, (width, height))

        prediction = model.predict(resizedFace)
        if prediction[1] < 800:
            cv2.putText(img, "%s - %.0f" % (names[prediction[0]], prediction[1]), (x-10,y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(img, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
            if (cnt > 100):
                print("Unknown Person")
                cv2.imwrite("unknown.jpg", img)
                cnt = 0
    cv2.imshow("Face Recognition", img)
    key = cv2.waitKey(10)
    if key == 27:
        break
cv2.destroyAllWindows()
