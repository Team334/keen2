import cv2
import numpy as np
import compat

# returns writer for video
def initOutputVideo(w, h, name='out_video.avi'):
    cv_version = cv2.__version__.split('.')[0]

    fourcc = compat.fourcc('M', 'J', 'P', 'G')
    
    writer = cv2.VideoWriter()
    print('open writer', writer.open(name, fourcc, 25, (w, h)))
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
#   *textToDisplay = map of values you want to display
#
# Returns image with values
def putValuesOnImage(image, data):
    count = 0
    for key, value in data.items():
        cv2.putText(image, "{} {}".format(key, value), (10, 40+30*count), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
        count += 1
    
    return image

# Sends an image through VisionTables
#
# Arguements:
#   *vision_table = initialized vision table
#   *image = image to send
def sendImage(vision_table, image):
    img_str = cv2.imencode('.jpg', image)[1].tostring()
    vision_table.putRaw('post_image', img_str)

