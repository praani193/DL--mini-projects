import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures

url = 'http://192.168.81.99/cam-hi.jpg'  # Ensure this URL is correct and accessible
im = None

def fetch_image():
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        return cv2.imdecode(imgnp, -1)
    except Exception as e:
        print(f"Error fetching image: {e}")
        return None

def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        im = fetch_image()
        if im is not None:
            cv2.imshow('live transmission', im)
        else:
            print("Failed to fetch image in live transmission")

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        im = fetch_image()
        if im is not None:
            bbox, label, conf = cv.detect_common_objects(im)
            im = draw_bbox(im, bbox, label, conf)
            cv2.imshow('detection', im)
        else:
            print("Failed to fetch image in detection")

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use ThreadPoolExecutor for I/O bound tasks
        f1 = executor.submit(run1)
        f2 = executor.submit(run2)
