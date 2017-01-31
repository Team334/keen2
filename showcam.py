import cv2
import numpy as np
import time
import utils.sortcoords

from networktables import NetworkTables
NetworkTables.initialize(server='10.3.34.22')

VISION_TABLE = NetworkTables.getTable('vision')

cap = cv2.VideoCapture(0)
fourcc = cv2.cv.FOURCC('M', 'P', 'E', 'G')
writer = cv2.VideoWriter()
print('open writer', writer.open('output_video.avi', fourcc, 25, (640, 480)))

prev_time = time.time()
times = np.zeros((30,))

def process_frame(frame):
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #thresh = cv2.inRange(hsv, np.array([30, 200, 100]), np.array([60, 255, 255]))

    #eroded = cv2.erode(thresh, (5,5), iterations=4)
    #dilated = cv2.dilate(eroded, (5,5), iterations=1)
    
    frame = cv2.medianBlur(frame,5) 
    blue, green, red = img[:,:,0], img[:,:,1], img[:,:,2]
    blue_scale, red_scale = .9, .9
    img_combo = green - blue * blue_scale - red * red_scale # highlights how green something is
    img_combo[img_combo < 0] = 0 # turn negatives to 0 to avoid wrapping 
    img_combo = img_combo.astype(dtype=np.uint8)
    _, thresh = cv2.threshold(img_combo,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    in_copy = frame.copy()

    #contours, h = cv2.findContours(dilated.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contours, h = cv2.findContours(thresh.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    good_contours = None
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect)
        #box = cv2.boxPoints(rect)
        cv2.drawContours(in_copy, [np.int0(box)], 0, (255, 0, 255), thickness=3)

        if good_contours is None:
            good_contours = c
        else:
            good_contours = np.concatenate((good_contours, c),axis=0)

    if good_contours is not None:
        hull = cv2.convexHull(good_contours)
        cv2.drawContours(in_copy, [hull], 0, (0, 0, 255), thickness=3)

        M = cv2.moments(hull)
        center_point = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        # approx = cv2.approxPolyDP(hull, 0.06*cv2.arcLength(hull,True), True)
        # if len(approx) == 4:
        #     quad_verts = np.array([[0, 0], [315,0], [315,150], [0,150]],dtype=np.float32).reshape((4, 1, 2))
        #     approx = np.array(approx, dtype=np.float32)

        #     approx = utils.sortcoords.sort_center(approx, center_point)
        #     approx = np.roll(approx, -1, axis=0)
        #     T = cv2.getPerspectiveTransform(approx, quad_verts)
        #     warped = cv2.warpPerspective(frame.copy(), T, dsize=(315,150))
        #     #cv2.imshow('warp', warped)

        x_offset = center_point[0] - (in_copy.shape[1]/2.0)
        min_x = hull[:,:,0].min()
        max_x = hull[:,:,0].max()

        half_length_px = max_x - center_point[0]

        width_approx = hull[:,:,0].max() - hull[:,:,0].min()
        height_approx = hull[:,:,1].max() - hull[:,:,1].min()

        area = cv2.contourArea(hull, True)

        if width_approx > height_approx:
            px_offset = center_point[0] + (half_length_px / (10.25/2)) * 8 - in_copy.shape[1]/2
        else:
            px_offset = center_point[0] + (width_approx/2.0)*3.8 - in_copy.shape[1]/2
            area = height_approx * (width_approx*(10.25/2.0))

        mid_x = (hull[:,:,0].min() + hull[:,:,0].max())/2.0
        canvas = np.zeros(in_copy.shape, dtype=np.uint8)
        pts = hull.reshape((1, -1, 2)).astype(np.int32)

        cv2.fillPoly(canvas, pts, (255,255,255))

        left_count = np.count_nonzero(canvas[:,:int(mid_x)])
        right_count = np.count_nonzero(canvas[:,int(mid_x):])
        total_count = np.count_nonzero(canvas)
        skew = float(left_count - right_count) / total_count
        VISION_TABLE.putNumber('skew', skew)

        cv2.putText(in_copy, "Skew %.3f"%skew, (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(in_copy, "Area %.3f"%area, (10, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        cv2.putText(in_copy, "Offset %.3f"%px_offset, (10, 135), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        print("Skew: %.3f" % skew)
        cv2.circle(in_copy, (int(px_offset+in_copy.shape[1]/2), int(in_copy.shape[0]/2)), 7, (255,0,0), thickness=-1)
        print("putting offset,area: ", px_offset, area)
        VISION_TABLE.putNumber('x_offset', px_offset)
        VISION_TABLE.putNumber('area', area)

    global times
    times = np.roll(times, -1)
    times[-1] = time.time() - prev_time
    fps = '%.1f FPS' % (1 / np.mean(times))
    cv2.putText(in_copy, fps, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

    writer.write(in_copy)

    img_str = cv2.imencode('.jpg', in_copy)[1].tostring()
    VISION_TABLE.putRaw('post_image', img_str)

    return in_copy

while True:
    ret, frame = cap.read()

    # rotate once clockwise
    #frame = np.rot90(frame, k=-1)

    if frame is None:
        print("NO FRAME")

    output = process_frame(frame)

    #cv2.imshow('processed', output)

    prev_time = time.time()
    if cv2.waitKey(10)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
