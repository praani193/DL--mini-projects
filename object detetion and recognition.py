import tkinter as tk
from tkinter import simpledialog, messagebox
import sys
import os
import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and the configuration files
net = cv2.dnn.readNetFromCaffe("C:/Users/Lenovo/OneDrive/Desktop/intern/mobilenet_iter_73000.caffemodel", "C:/Users/Lenovo/OneDrive/Desktop/intern/deploy.caffemodel")

# Class labels MobileNet SSD supports (20 classes)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def object_detect():
    try:
        folder_name = simpledialog.askstring("Input", "Enter the folder name:")
        if folder_name:
            dataset = "dataset1"
            name = folder_name
            path = os.path.join(dataset, name)
            print(os.path.isdir(path))
            if not os.path.exists(path):
                os.makedirs(path)

            cam = cv2.VideoCapture(0)
            count = 1
            while count < 301:
                ret, frame = cam.read()
                if not ret:
                    print("Failed to capture image")
                    break

                (h, w) = frame.shape[:2]
                # Preprocessing the frame for MobileNet SSD
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # Loop over detections
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > 0.2:  # Only process detections with confidence greater than 0.2
                        idx = int(detections[0, 0, i, 1])
                        label = CLASSES[idx]

                        # Extract bounding box dimensions
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Draw the prediction on the frame
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        label_text = f"{label}: {confidence:.2f}"
                        cv2.putText(frame, label_text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)
                        print(f"Object Detected: {label} - Confidence: {confidence:.2f}")

                        # Save the object image
                        obj_img = frame[startY:endY, startX:endX]
                        cv2.imwrite(f"{path}/{label}_{count}.jpg", obj_img)
                        count += 1

                cv2.imshow("Object Detection", frame)
                key = cv2.waitKey(10)
                if key == 27:  # ESC key to stop
                    break

            print("Completed Object Detection")
            cam.release()
            cv2.destroyAllWindows()
        else:
            messagebox.showwarning("Input Error", "Folder name cannot be empty!")
    except Exception as e:
        print(f"Error occurred[Detect]: {str(e)}")


def object_recognize():
    try:
        datasets = 'dataset1'
        print('Training...')

        (images, labels, names, id) = ([], [], {}, 0)

        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                print("Subject Path:", subjectpath)
                for filename in os.listdir(subjectpath):
                    path = os.path.join(subjectpath, filename)
                    print("Image Path:", path)
                    label = id
                    image = cv2.imread(path)
                    if image is None:
                        print("Error loading image:", path)
                        continue
                    print("Image shape:", image.shape)
                    images.append(image)
                    labels.append(int(label))
                id += 1

        # Placeholder for object recognition implementation (This requires custom model training)
        print(f"Training done with {len(images)} images")

        # Using the pre-trained model for recognition (Example: Comparing new objects with the existing dataset)
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to capture image")
                break

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # Loop over detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"Recognized: {label} - Confidence: {confidence:.2f}")

            cv2.imshow("Object Recognition", frame)
            key = cv2.waitKey(10)
            if key == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred[Recognize]: {str(e)}")


root = tk.Tk()
root.title("Object Detection & Recognition")
root.geometry("400x200")
label = tk.Label(root, text="Choose an Option", font=("Arial", 16))
label.pack(pady=20)
detect_button = tk.Button(root, text="Object Detect", command=object_detect, width=20, height=2)
detect_button.pack(pady=10)
recognize_button = tk.Button(root, text="Object Recognition", command=object_recognize, width=20, height=2)
recognize_button.pack(pady=10)
root.mainloop()
