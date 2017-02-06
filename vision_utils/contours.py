import cv2
import numpy as np

# compute intersection over union (IOU) of two contours.
# IOU = (area of intersection) / (area of union)
#
# returns: (area of intersection, area of union)
def iou_frac(c1, c2):
    max_x = max(c1[:,:,0].max(), c2[:,:,0].max())
    max_y = max(c1[:,:,1].max(), c2[:,:,1].max())
    # just in case there is some float business going on
    max_x = int(max_x + 1)
    max_y = int(max_x + 1)

    canvas1 = np.zeros((max_x, max_y), dtype=np.uint8)
    canvas2 = canvas1.copy()
    cv2.fillPoly(canvas1, [c1], 1) 
    cv2.fillPoly(canvas2, [c2], 1)

    canvas = canvas1 + canvas2

    intersection = np.sum(canvas == 2)
    union = np.sum(canvas != 0)
    return intersection, union

# compute intersection over union (IOU) of two contours.
# IOU = (area of intersection) / (area of union)
def iou(c1, c2):
    i, u = iou_frac(c1, c2)
    return float(i) / u

# compute the center of contour. returns (cx, cy)
def center(contour):
    M = cv2.moments(contour)
    return M["m10"]/M["m00"], M["m01"]/M["m00"]

# sort a list of points in clockwise order around a center
#
# arguments:
#   * pts_list: the points to be sorted, stored as a 2D list:
#               [[x1, y1], [x2, y2], ...]
#   * center: the center point: [cx, cy]
#
# thanks to: http://stackoverflow.com/a/6989383/2498956
def sort_pts(pts_list, center):
    def cmp_pts(a, b):
        if a[0] - center[0] >= 0 and b[0] - center[0] < 0:
            return 1
        if a[0] - center[0] < 0 and b[0] - center[0] >= 0:
            return -1
        if a[0] - center[0] == 0 and b[0] - center[0] == 0:
            if a[1] - center[1] >= 0 or b[1] - center[1] >= 0:
                return cmp(a[1], b[1])
            return cmp(b[1], a[1])

        det = (a[0] - center[0])*(b[1] - center[1]) - (b[0] - center[0])*(a[1] - center[1])
        if det < 0:
            return 1
        if det > 0:
            return -1

        d1 = (a[0] - center[0])**2 + (a[1] - center[1])**2
        d2 = (b[0] - center[0])**2 + (b[1] - center[1])**2
        return cmp(d1, d2)

    sorted_pts = sorted(pts_list, cmp=cmp_pts)
    return sorted_pts

# sort a contour so that its points are in clockwise order
#
# arguments:
#   * contour: a numpy array of points with shape (n_points, 1, 2)
def sort_contour(contour):
    return sort_contour_center(contour, center(contour))

# sorts a contour so that its points are in clockwise order
#
# arguments:
#   * contour: a numpy array of points with shape (n_points, 1, 2)
#   * center: the center of the contour
def sort_contour_center(contour, center):
    pts = list(contour.reshape((-1, 2)))

    sorted_pts = sort_pts(pts, center)
    ret = np.array(sorted_pts).reshape((-1, 1, 2))
    return ret
