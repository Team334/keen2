import numpy as np
import cv2

# convert a hexadecimal color to a BGR color array
def hex2bgr(h):
    b = h & 0xFF
    g = (h >> 8) & 0xFF
    r = (h >> 16) & 0xFF
    return np.array([b, g, r])

# predefined colors in BGR
MAROON  = hex2bgr(0x800000)
RED     = hex2bgr(0xFF0000)
ORANGE  = hex2bgr(0xFFA500)
YELLOW  = hex2bgr(0xFFFF00)
OLIVE   = hex2bgr(0x808000)
GREEN   = hex2bgr(0x008000)
PURPLE  = hex2bgr(0x800080)
FUCHSIA = hex2bgr(0xFF00FF)
LIME    = hex2bgr(0x00FF00)
TEAL    = hex2bgr(0x008080)
AQUA    = hex2bgr(0x00FFFF)
BLUE    = hex2bgr(0x0000FF)
NAVY    = hex2bgr(0x000080)
BLACK   = hex2bgr(0x000000)
GRAY    = hex2bgr(0x808080)
SILVER  = hex2bgr(0xC0C0C0)
WHITE   = hex2bgr(0xFFFFFF)

# convert a color into another colorspace
#
# arguments:
#   * c: the color to convert. Can be a list, numpy array, or tuple.
#   * code: the opencv color conversion code, e.g. cv2.COLOR_BGR2HSV
def cvtPixel(c, code):
    img = np.array(c)
    if img.dtype == np.float64:
        img = img.astype(np.float32, copy=False)
    elif img.dtype != np.float32:
        img = img.astype(np.uint8, copy=False)
    img = img.reshape((1, 1, -1))
    converted = cv2.cvtColor(img, code)
    return converted[0][0]
