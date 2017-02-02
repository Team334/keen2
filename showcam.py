import cv2, sys, time
import numpy as np
import debug, compat, process

from networktables import NetworkTables
NetworkTables.initialize(server='10.3.34.22')

VISION_TABLE = NetworkTables.getTable('vision')

if len(sys.argv) != 2:
    print ("Arguments required: mode")
    sys.exit(-1)
mode = sys.argv[1]

cap = cv2.VideoCapture(0)
writer = None
if mode == "DEBUG":
    writer = debug.initOutputVideo(w=640, h=480)
    #writer = debug.initOutputVideo(w=1280, h=720)

prev_time = time.time()
times = np.zeros((30,))

def process_frame(frame):
    # stores any values you want to print
    textToDisplay = {}

    in_copy = frame.copy()
    
    thresh = process.adaptiveGreenThreshold(in_copy)
    
    contours = compat.findContours(thresh.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    good_contours = process.filterContours(contours)
    cv2.drawContours(in_copy, [good_contours], 0, (255, 0, 255), thickness=3)
    
    if good_contours is not None:
        hull = cv2.convexHull(good_contours)
        cv2.drawContours(in_copy, [hull], 0, (0, 0, 255), thickness=3)

        px_offset, area = process.getOffsetAndArea(in_copy, hull)
        cv2.circle(in_copy, (int(px_offset+in_copy.shape[1]/2), int(in_copy.shape[0]/2)), 7, (255,0,0), thickness=-1)
        
        skew = process.getSkew(in_copy, hull)
        
        textToDisplay.update({"Skew":skew, "Area":area, "Offset":px_offset})

        print("Skew: %.3f" % skew)
        print("putting offset,area: ", px_offset, area)
        VISION_TABLE.putNumber('x_offset', px_offset)
        VISION_TABLE.putNumber('area', area)
        VISION_TABLE.putNumber('skew', skew)

    global times
    times = np.roll(times, -1)
    times[-1] = time.time() - prev_time
    fps = (1 / np.mean(times))
    textToDisplay["fps"] = fps

    if mode == "DEBUG":
        debug.putValuesOnImage(in_copy, textToDisplay)
        debug.writeToVideo(in_copy, writer)
        debug.sendImage(VISION_TABLE, in_copy)

    return in_copy

while True:
    ret, frame = cap.read()
    if frame is None:
        print("NO FRAME")

    output = process_frame(frame)

    cv2.imshow('processed', output)
    prev_time = time.time()
    if cv2.waitKey(10)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
