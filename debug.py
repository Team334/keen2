import cv2
import numpy as np

# returns writer for video
def initOutputVideo(w, h, name='out_video.avi'):
    cv_version = cv2.__version__.split('.')[0]
    print(cv_version)
    fourcc = None
    if cv_version == "2":
        fourcc = cv2.cv.FOURCC('M', 'P', 'E', 'G')
    elif cv_version == "3":
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
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
#   *dict = map of values you want to display
#
# Returns image with values
def putValuesOnImage(image, dict):
    count = 0
    for key in dict:
        cv2.putText(image, "{} {:f}".format(key, dict[key]), (10, 40+30*count), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255))
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

