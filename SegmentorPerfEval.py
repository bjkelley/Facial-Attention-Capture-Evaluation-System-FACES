from SegmentorClass import Segmentor
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import sys, os
import pickle
import gc

if sys.argv[1] == "-h" or sys.argv[1] == "-H" :
    print("Segmentor Class Evaluation Script"
          "     -h : This help menu"
          "     -E : Calculate performance of Segmentor class on WIDER FACE"
          "     -D : Draw Prefermance Chart form Pefsave.p")
elif  sys.argv[1] == "-E":
    print("Evaluating")

    ValsFile = 'ValidationDataSet/wider_face_val_bbx_gt.txt'
    imRoot = 'ValidationDataSet/images/'

    impls ={"Haar": Segmentor(impl="Haar"),
            "Yolo": Segmentor(impl="Yolo"),
            "faced": Segmentor(impl="faced")}

    # impls = {"Yolo": Segmentor(impl="Yolo")}

    scores = {}


    def bb_intersection_over_union(boxA, boxB):
        # source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    # boxA = [83-15, 304-15, 83+8+15, 304+7+15]
    # boxB =  [66, 289, 66+24, 289+24]
    # print("god {}".format(bb_intersection_over_union(boxA, boxB)))

    fullstarttime = time.time()


    # =======================Warning=======================
    # The following code is terrible, save yourself and don't look though it


    for impl in impls:

        numPhotos = 0
        sum_eval_score = 0
        total_time = 0

        with open(ValsFile, 'r') as vf:
            numlines = len(vf.readlines())
            vf.seek(0) #reset head


            for _ in range(numlines-1):
                numPhotos += 1

                #get filename and number of spaces
                path = vf.readline()
                if path == '': break
                path = imRoot + path[:-1] # :-1 removes return carriage
                frame = cv2.imread(path)
                n = int(vf.readline())
                true_faces = []

                for i in range(n):
                    loc = vf.readline()
                    loc = loc.split(' ')[:-1] # :-1 gets rid of return carriage
                    loc = [int(L) for L in loc] # convert ot int

                    buff = 15
                    true_faces.append([loc[0]- buff, loc[1]- buff, loc[2]+ buff, loc[3]+ buff]) # x, y, w ,h

                    # cv2.rectangle(frame, (loc[0] - buff, loc[1] - buff), (loc[0] + loc[2] + buff, loc[1] + loc[3] + buff), (0, 255, 0), 2)
                    # image = cv2.putText(frame, str(i), (loc[0]- buff, loc[1]- buff), cv2.FONT_HERSHEY_SIMPLEX ,
                    #                     1, (0, 255, 0) , 1, cv2.LINE_AA)
                    # print(loc)

                np.array(loc)

                #Model Eval
                sys.stdout = open(os.devnull, 'w') # get rid of some prints in class
                if impl == "Yolo":
                    # start_Model = 0; eval_faces = [[1,1,5,5]] #Debug
                    tmp = Segmentor(impl="Yolo")
                    start_Model = time.time()
                    eval_faces ,_ = tmp.Segment(frame) # because Darknet doenst like changing dims
                    total_time += time.time() - start_Model
                    gc.collect()
                else:
                    start_Model = time.time()
                    eval_faces ,_ = impls[impl].Segment(frame)
                    total_time += time.time() - start_Model
                sys.stdout = sys.__stdout__ # re enable prints

                #IOU
                # print(eval_faces)
                # print(true_faces[11])
                # print(true_faces[12])
                # this is n^2 but im lazy so  ¯\_(ツ)_/¯
                # this evaluate bounding for each face in eval faces to every known face. Score is max of IOU
                sum_score = 0
                for face in eval_faces:
                    # cv2.rectangle(frame, (face[0] - buff, face[1] - buff), (face[0] + face[2] + buff, face[1] + face[3] + buff),
                    #               (255, 0, 0), 1)

                    # convert from x,y,w,h to x,y,x,y
                    face[0] = face[0] - buff
                    face[1] = face[1] - buff
                    face[2] = face[0] + face[2] + 2*buff
                    face[3] = face[1] + face[3] + 2*buff

                    face_score = []
                    for face2 in true_faces:
                        # convert from x,y,w,h to x,y,x,y
                        face2[2] = face2[0] + face2[2]
                        face2[3] = face2[1] + face2[3]

                        face_score.append(bb_intersection_over_union(face, face2))

                    maxScore = np.max(np.array(face_score))
                    # print("Something {}".format(maxScore))
                    sum_score += maxScore
                eval_score = sum_score/len(true_faces)
                sum_eval_score += eval_score



                # cv2.imshow('pic', frame)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     break

                # break

                if numPhotos % 100 == 0:
                    print("Evaluating {} implantation, {} photos evaluated, Time elapsed: {}".format(impl, numPhotos, time.time()-fullstarttime))

        scores[impl] = (total_time/numPhotos, sum_eval_score/numPhotos)

        print("Saving...")
        pickle.dump(scores, open("Pefsave.p", "wb"))
        print("... Done Saving")

    print(scores)

elif sys.argv[1] == "-D":
    print("Drawing Fig")
    scores = pickle.load(open("Pefsave.p", "rb"))

    marks =  [('r', "D"), ("g","*"), ("m","X")]

    for i, impl in enumerate(scores.keys()):
        x, y = scores[impl]
        print(x)
        print(y)
        plt.scatter(x, y, 100, c=marks[i][0], alpha=0.5, marker=marks[i][1],
                    label=impl)

    plt.legend()
    plt.xlabel("Average Evaluation Time (s)")
    plt.ylabel("Average IoU")

    plt.show()
else:
    print("Unknown Arg")