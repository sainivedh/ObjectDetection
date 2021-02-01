import numpy as np
import cv2
from scipy.spatial import distance as dist
import argparse
import imutils
import cv2
import os

from VehicleCollisionAlert import vehicle_collision_alert_config as config
from VehicleCollisionAlert.detections_alg import detections

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to output video  file")
#args = vars(ap.parse_args(["--input","D:\Books\study\python\Object_Detection\VehicleDetectionAlert\Traffic_cropped.mp4","--output","output_correct.avi"]))
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

while True:
    (grabbed, frame) = vs.read()
    #print('helooo')
    if not grabbed:
        print('exit')
        break

    frame = imutils.resize(frame, 700)
    vehiclesIdx = [LABELS.index("bicycle"), LABELS.index("car"), LABELS.index("motorbike"), LABELS.index("bus"), LABELS.index("truck")]
    results = detections(frame, net, ln, vehiclesIdx)

    violate = set()

    if len(results) >= 2:
        centroids = np.array([x[2] for x in results])
        D = dist.cdist(centroids, centroids)

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i,j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):

        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0,255,0)

        if i in violate:
            color = (0,0,255)
            text = "Collision Alert"
            cv2.putText(frame, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 1)
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #cv2.circle(frame, (cX, cY), 5, color, 1)

        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25, 
                                        (frame.shape[1], frame.shape[0]), True)
        
        if writer is not None:
            writer.write(frame)


os.system('ffmpeg -i output_correct.avi -vf "setpts=0.75*PTS" traffic_collisionAlert.mp4')