import cv2
import time
import numpy as np
from seg_utils import *
from loader_class import *
from FacedSegmentor2 import FaceDetector

model_cfg = 'yolov3-face.cfg'
model_weights = 'yolov3-wider_16000.weights'
cascPath = "haarcascade_frontalface_default.xml"
# yolov3-wider_16000.weights is too big to push to github for some reason
buff = 10 # oversize drawn rectangle for viewing


class Segmentor:
    def __init__(self, impl="faced"):
        imples = {
                  "Haar": (self.HaarSegment, self.HaarInit),
                  "Yolo": (self.YoloSegment, self.YoloInit),
                  "faced": (self.facedSegment, self.facedInit)
                }

        self.initimple = imples[impl][1]() #run initilizer
        self.Segment = imples[impl][0]
        self.output_dim = (128,128)
        self.frame = []

    def facedInit(self):
        self.face_detector = FaceDetector()

    def facedSegment(self, image):
        """
        Use YOLO segmentor to get facial images

        :return:
        List of arrays: [X, Y, W, H] and list of np arrays of cut images

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.predict(image) #get bounding boxes
        Face_im = []
        for (x, y, w, h, _) in bboxes:
            cut_face = image[y - buff:y + h + buff, x - buff:x + w + buff]
            cut_face = self.resize(cut_face)
            if(cut_face is not None and cut_face.all() != None):
                Face_im.append(cut_face)

        #convert from center of rectangle to top left corner
        faces = [[x - w // 2, y - h // 2, w, h] for (x, y, w, h, _) in bboxes]

        return faces, Face_im

    def YoloInit(self):
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



    def YoloSegment(self, image):
        """
        Use YOLO segmentor to get facial images

        :return:
        List of arrays: [X, Y, W, H] and list of np arrays of cut images

        """

        Face_im = []

        image = image[0:((image.shape[0] // 32) * 32), 0:((image.shape[1] // 32) * 32), :]

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(image, 1 / 255, (image.shape[0] , image.shape[1]),
                                     [0, 0, 0], 1, crop=False)

        print("yolo")
        print(image.shape)


        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(get_outputs_names(self.net))

        # Remove the bounding boxes with low confidence
        faces = post_process(image, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        for (x, y, w, h) in faces:
            cut_face = image[y - buff:y + h + buff, x - buff:x + w + buff]
            cut_face = self.resize(cut_face)
            if(type(cut_face) != None):
                Face_im.append(cut_face)

        print(faces)
        # if (len(Face_im) > 0):
        #     print(Face_im[0].shape)
        return faces, Face_im


    def refined_box(self, left, top, width, height):
        right = left + width
        bottom = top + height

        original_vert_height = bottom - top
        top = int(top + original_vert_height * 0.15)
        bottom = int(bottom - original_vert_height * 0.05)

        margin = ((bottom - top) - (right - left)) // 2
        left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

        right = right + margin

        return left, top, right, bottom

    #initializes Haar classifier
    def HaarInit(self):
        self.faceCascade = cv2.CascadeClassifier(cascPath)


    def HaarSegment(self, image):
        """
        Use Haar segmentor to get facial images

        :return:
        List of arrays: [X, Y, W, H] and list of np arrays of cut images

        """
        Face_im = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(5, 5)
        )
        for (x, y, w, h) in faces:
            cut_face = image[y - buff:y + h + buff, x - buff:x + w + buff]
            cut_face = self.resize(cut_face)
            if(type(cut_face) != None):
                Face_im.append(cut_face)

        return faces, Face_im

    #resizes input image to the necessary output dimension
    def resize(self, im):
        try:
            assert(im.size != 0)
            resized_face = cv2.resize(im, self.output_dim, interpolation=cv2.INTER_AREA)
        except AssertionError:
            return None
        return resized_face

    def Segment(self, image):
        """
        Use Haar segmentor to get facial images

        :return:
        List of arrays: [X, Y, W, H] and list of np arrays of cut images

        """
        pass


if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    model = ReadyModel('generalizedSVM')

    segmentor = Segmentor()

    while True:
        ret, segmentor.frame = video_capture.read() #read frame from video
        faces, cuts = segmentor.Segment(segmentor.frame) #detect faces in frame
        print("Loading sample...", end="\r")
        print(f"Found {len(cuts)} faces.")
        for i in range(len(cuts)):
                sample = cuts[i]
                print(sample.shape)
                print("Making single prediction...", end="\r")
                startTime = time.time()
                singleResult = model.classify(sample) #emotional classification
                predictTime = time.time() - startTime
                print(f"Processed in {predictTime} seconds.")
                print(singleResult)

        #draw rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(segmentor.frame, (x - buff, y - buff), (x + w + buff, y + h + buff), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', segmentor.frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
