import cv2
import numpy as np
import compat
import time

# Finds green targets in axm image
# Green channel - Blue channel - Red channel (Team 900 formula)
#   *Green only = high
#   *Green + low red/blue = high
#   *Green + high red/blue = low
#   *Green + high red + high blue = low
#
# Arguments:
#   *image = image to threshold
#
# returns thresholded image
def adaptiveGreenThreshold(image):
    #image = cv2.medianBlur(image,5) 
    blue, green, red = image[:,:,0], image[:,:,1], image[:,:,2]
    blue_scale, red_scale = .9, .9
    img_combo = green - blue * blue_scale - red * red_scale # highlights how green something is
    img_combo[img_combo < 0] = 0 # turn negatives to 0 to avoid wrapping 
    img_combo = img_combo.astype(dtype=np.uint8)
    _, thresh = cv2.threshold(img_combo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    eroded = cv2.erode(thresh, (3,3), iterations=2)
    dilated = cv2.dilate(eroded, (3, 3), iterations=2)

    return dilated

# Finds target possibilities
#
# Arguments:
#   *contours = all contours to look through
# 
# returns filtered contours
def filterContours(contours):
    areaThreshold = 300
    good_contours = None
    for c in contours:
        if cv2.contourArea(c) < areaThreshold:
           continue
        rect = cv2.minAreaRect(c)
        box = compat.boxPoints(rect)
        if good_contours is None:
            good_contours = c
        else:
            good_contours = np.concatenate((good_contours, c), axis=0)
    
    return good_contours

# Calculate offset = distance from centroid of contours to center of image + calibration
#
# Arguments:
#   *image = image containing center_point
#   *hull = shape containing all the contours
#   
# returns offset and area
def getOffsetAndArea(image, hull):
    M = cv2.moments(hull)
    contour_center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

    min_x = hull[:,:,0].min() 
    max_x = hull[:,:,0].max()
    
    half_length_px = max_x - contour_center[0]

    width_approx = hull[:,:,0].max() - hull[:,:,0].min() 
    height_approx = hull[:,:,1].max() - hull[:,:,1].min()
   
    px_offset, area = None, None
    isCutOff = width_approx < height_approx
    if not isCutOff:
        px_offset = contour_center[0] + (half_length_px / (10.25/2)) * 8 - image.shape[1]/2
        area = cv2.contourArea(hull, True)
    # IMAGE IS CUT OFF
    # APPROXIMATIONS ARE MADE
    else:
        px_offset = contour_center[0] + (width_approx/2.0)*3.8 - image.shape[1]/2
        area = height_approx * (width_approx*(10.25/2.0))
    
    return px_offset, area

# Caclulate how tilted(skew) target is compared to perfect rectangle
#
# returns skew of target
def getSkew(image, hull):
    mid_x = (hull[:,:,0].min() + hull[:,:,0].max())/2.0
    canvas = np.zeros(image.shape, dtype=np.uint8)
    pts = hull.reshape((1, -1, 2)).astype(np.int32)
    
    cv2.fillPoly(canvas, pts, (255,255,255))

    left_count = np.count_nonzero(canvas[:,:int(mid_x)])
    right_count = np.count_nonzero(canvas[:,int(mid_x):])
    total_count = np.count_nonzero(canvas)
    skew = float(left_count - right_count) / total_count

    return skew

# Find angle from center to pixel
# OFFSET / total_width = ? / FOV
#
# Arguments:
#   *image = image containing pixel
#   *pixel = point on image
#
# returns angle in degrees
def getHorizontalAngleToPixel(image, x_offset):
    img_width = image.shape[1]
    cam_FOV = 47 # aspect ratio = 4:3, diagonal = 57 degrees

    return (x_offset / img_width) * cam_FOV

