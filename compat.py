# OPENCV FUNCTIONS WITH COMPATIBILITY FOR BOTH OPENCV2 AND OPENCV3

import cv2
import numpy as np

def findContours(*args, **kwargs):
    cv_version =  cv2.__version__.split(".")[0]
    if cv_version == "2":
        return cv2.findContours(*args, **kwargs)[0]
    elif cv_version == "3":
        return cv2.findContours(*args, **kwargs)[1]

def boxPoints(rect):
    cv_version =  cv2.__version__.split(".")[0]
    if cv_version == "2":
        return cv2.cv.BoxPoints(rect)
    elif cv_version == "3":
        return cv2.boxPoints(rect)

def fourcc(*args):
    cv_version =  cv2.__version__.split(".")[0]
    if cv_version == "2":
        return cv2.cv.FOURCC(*args)
    elif cv_version == "3":
        return cv2.VideoWriter_fourcc(*args)
