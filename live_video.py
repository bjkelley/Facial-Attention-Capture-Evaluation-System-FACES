import numpy as np
import cv2
from multiprocessing import Queue
from threading import Thread
from loader_class import *
from SegmentorClass import *

def convertToGrayscale():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def segmentAndPredict(segmentor, frame, queue):
    print("START THREAD")
    faces, cuts = segmentor.Segment(frame)

    if len(cuts) > 0:
            print("Loading single sample...", end="\r")
            sample = cuts[0]
            print(sample.shape)
            print("Making single prediction...", end="\r")
            startTime = time.time()
            singleResult = model.classify(sample)
            predictTime = time.time() - startTime
            print(f"Processed in {predictTime} seconds.")
            print(singleResult)
    
    queue.put(faces)

if __name__ == "__main__":    
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = ReadyModel('generalizedSVM')
    segmentor = Segmentor('Yolo')

    IMG_PROCESS_TIME = 2.5
    thread = None
    faces = []
    facesQueue = Queue()

    video_capture = cv2.VideoCapture(0) # Begin capturing video from camera

    lastImageProcess = time.time()
    while True:
        ret, frame = video_capture.read()

        if time.time() - lastImageProcess > IMG_PROCESS_TIME:    # Update image segmentation and emotional classification at a constant rate
            if (thread):
                thread.join()   # Make sure image processing has finished and get results
                faces = facesQueue.get()

            lastImageProcess = time.time()
            thread = Thread(target=segmentAndPredict, args=(segmentor, frame, facesQueue), daemon=True) # Start new image processing
            thread.start()

        for (x, y, w, h) in faces:  # Draw rectangle around each face
            cv2.rectangle(frame, (x - buff, y - buff), (x + w + buff, y + h + buff), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break