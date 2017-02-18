import cv2
import sys
import time

import numpy as np

import nt
import debug
import compat
import process
from vision_utils.fps import FPSCounter

if len(sys.argv) != 2:
    print ("Arguments required: mode")
    sys.exit(0)
mode = sys.argv[1]

VISION_TABLE = nt.initTable()

CAMERA_KEY = "camera"
GEAR_VALUE = "gear"
BOILER_VALUE = "boiler"

# automatically updates when value is changed
cam = VISION_TABLE.getAutoUpdateValue(CAMERA_KEY, GEAR_VALUE)

gearCap = cv2.VideoCapture(0)
boilerCap = cv2.VideoCapture(1)
cap = gearCap # default camera

writer = None
if mode == "DEBUG":
    writer = debug.initOutputVideo(w=640, h=480)
    #writer = debug.initOutputVideo(w=1280, h=720)

fps = FPSCounter()

def process_frame(frame):
    # stores any values you want to print or send
    data = {}

    in_copy = frame.copy()
    thresh = process.adaptiveGreenThreshold(in_copy)
    contours = compat.findContours(thresh.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    good_contours = process.filterContours(contours)

    # found a target
    if good_contours is not None:
        data["found"] = True

        hull = cv2.convexHull(good_contours)
        #cv2.drawContours(in_copy, [hull], 0, (0, 0, 255), thickness=3)

        px_offset, area = process.getOffsetAndArea(in_copy, hull)
        cv2.circle(in_copy, (int(px_offset+in_copy.shape[1]/2), int(in_copy.shape[0]/2)), 7, (255,0,0), thickness=-1)
        angle = process.getHorizontalAngleToPixel(in_copy, px_offset)
        #skew = process.getSkew(in_copy, hull)

        data.update({"area":area, "x_offset":px_offset, "angle":angle})

        print("angle: ", angle)
        print("putting offset,area: ", px_offset, area)
    else:
        data["found"] = False

    nt.sendData(VISION_TABLE, data)

    fps.got_frame()
    data["fps"] = fps.fps()
    print("FPS = ", fps.fps())
   
    data["camera"] = cam.value

    if mode == "DEBUG":
        # draw
        cv2.drawContours(in_copy, [good_contours], 0, (255, 0, 255), thickness=3)
        cv2.circle(in_copy, (int(in_copy.shape[1]/2), int(in_copy.shape[0]/2)), 7, (0,255,0), thickness=-1)
        
        debug.putValuesOnImage(in_copy, data)
        debug.writeToVideo(in_copy, writer)
        #debug.sendImage(VISION_TABLE, in_copy)

    return in_copy

while True:
    nt.sendData(VISION_TABLE, {"running":True})

    # choose camera
    if cam.value == GEAR_VALUE:
        cap = gearCap
    elif cam.value == BOILER_VALUE:
        cap = boilerCap

    ret, frame = cap.read()
    if frame is None:
        print("NO FRAME")

    output = process_frame(frame)

    #cv2.imshow('processed', output)

    if cv2.waitKey(10)&0xFF==ord('q'):
        break

# clean everything up
gearCap.release()
boilerCap.release()
cv2.destroyAllWindows()
