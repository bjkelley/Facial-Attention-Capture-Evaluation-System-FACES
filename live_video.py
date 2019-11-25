import numpy as np
import cv2
from multiprocessing import Queue
from threading import Thread
from loader_class import *
from SegmentorClass import *

ATTENTIVE_COLOR = (0, 255, 0) # Green
INATTENTIVE_COLOR = (0, 0, 255) # Red 

EMOTION_COLORS = {  # BGR Values
    'neutral frontal': INATTENTIVE_COLOR,
	'joy': ATTENTIVE_COLOR,
	'sadness': INATTENTIVE_COLOR,
	'surprise': ATTENTIVE_COLOR,
	'anger': INATTENTIVE_COLOR,
	'disgust': INATTENTIVE_COLOR,
	'fear': INATTENTIVE_COLOR,
	'opened': ATTENTIVE_COLOR,
    'closed': INATTENTIVE_COLOR,
	'kiss': ATTENTIVE_COLOR
}

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
    faces, cuts = segmentor.Segment(frame)
    predictions = []

    if len(cuts) > 0:
        print("Loading samples...", end="\r")
        for sample in cuts:
            
            print(sample.shape)
            print("Making single prediction...", end="\r")
            startTime = time.time()
            prediction = model.classify(sample)
            predictTime = time.time() - startTime
            print(f"Processed in {predictTime} seconds.")
            print("PRED: ", prediction[0])
            predictions.append(prediction[0])
    
    queue.put(faces)
    queue.put(predictions)

if __name__ == "__main__":    
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = ReadyModel('generalizedSVM')
    segmentor = Segmentor('Yolo')
    BUFF = 15

    IMG_PROCESS_TIME = 2.5
    thread = None
    faces = []
    resultsQueue = Queue()

    video_capture = cv2.VideoCapture(0) # Begin capturing video from camera

    lastImageProcess = time.time()
    while True:
        ret, frame = video_capture.read()

        if time.time() - lastImageProcess > IMG_PROCESS_TIME:    # Update image segmentation and emotional classification at a constant rate
            if (thread):
                thread.join()   # Make sure image processing has finished and get results
                faces = resultsQueue.get()
                predictions = resultsQueue.get()

            lastImageProcess = time.time()
            thread = Thread(target=segmentAndPredict, args=(segmentor, frame, resultsQueue), daemon=True) # Start new image processing
            thread.start()

        for i in range(0, len(faces)):
            x, y, w, h = faces[i]  # Draw rectangle around each face
            color = EMOTION_COLORS[predictions[i][0][1]]
            cv2.rectangle(frame, (x - BUFF, y - BUFF), (x + w + BUFF, y + h + BUFF), color=color, thickness=2)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break