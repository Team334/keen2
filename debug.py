import cv2
import numpy as np

# returns writer for video
def initOutputVideo(w, h):
    fourcc = cv2.cv.FOURCC('M', 'P', 'E', 'G')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    writer = cv2.VideoWriter()
    print('open writer', writer.open('output_video.avi', fourcc, 25, (w, h)))
    return writer

# Writes an image to the video
#
# Arguments:
#   *image = image to write
#   *writer = videowriter
def writeToVideo(image, writer):
    writer.write(image)

# Displays values on image
#
# Arguments:
#   *image = image to put values on
#   *keys = names of values being displayed
#   *values = values being displayed
#
# Returns image with values
def putValuesOnImage(image, keys, values):
    for i in range(len(keys)):
        cv2.putText(image, "{} {:f}".format(keys[i], values[i]), (10, 40+30*i), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
    
    return image

# Sneds an image through VisionTables
#
# Arguements:
#   *VISION_TABLE = initialized vision table
#   *image = image to send
def sendImage(VISION_TABLE, image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    VISION_TABLE.putRaw('post_image', img_str)

