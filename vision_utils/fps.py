import numpy as np
import cv2
import time

# utility for measuring frames per second (FPS). Uses a
# rolling average to smooth measurements
class FPSCounter:

    # initialize an FPS counter with a rolling window length of
    # avg_len frames (30 frames by default)
    def __init__(self, avg_len=30):
        self.readings = np.zeros((avg_len,))
        self.n_readings = 0

        self.prev_time = time.time()
    
    # tell the counter that a frame was processed
    def got_frame(self):
        now = time.time()
        delta = now - self.prev_time
        self.prev_time = now
        self.readings = np.roll(self.readings, 1)
        self.readings[0] = delta

        if self.n_readings < self.readings.size:
            self.n_readings += 1

    # get the average FPS
    def fps(self):
        if self.n_readings == 0:
            return 0.0
        else:
            valid_readings = self.readings[:self.n_readings]
            return 1.0 / np.mean(valid_readings)

    # draw the average on an image
    #
    # arguments
    #   * img:       the image to draw on
    #   * pos:       the position (x,y) to draw the text at
    #   * prec:      the number of decimal points to display
    #   * size:      the text size
    #   * thickness: the text thickness
    #   * color:     the text color
    #   * fmt:       the text formatting string
    def draw_fps(self, img, pos=(10,30), prec=1, size=2, thickness=2, color=(255,255,255), fmt='%.1f FPS'):
        n = round(self.fps(), prec)
        text = fmt % n
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_PLAIN, size, color, thickness)
