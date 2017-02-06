# vision utils

Various utilities for OpenCV.

## Documentation

### `fps.py`

- **`class FPSCounter`**: utility for measuring frames per second (FPS) with a rolling average
  - `FPSCounter(avg_len=30)`: create a new FPS counter that averages `avg_len` measurements
  - `got_frame()`: tell the counter that a frame was processed
  - `fps()`: get the average FPS
  - `draw_fps(self, img, pos=(10,30), prec=1, size=2, thickness=2, color=(255,255,255), fmt='%.1f FPS')`: draw the FPS on an image
    - `img`: the image to draw on
    - `pos`: the position to draw the text at
    - `prec`: the number of decimal digits to display
    - `size`: the size of the text
    - `thickness`: the thickness of the text
    - `color`: the color of the text
    - `fmt`: the format string for the text. Must format a float.
    
### `colors.py`

Color-related utilities

- `hex2bgr(h)`: converts a hexadecimal color into a BGR numpy array. E.g. `0x00FF01` -> `[1, 255, 0]`
- `cvtPixel(color, code)`: converts a color into another color space.
  - `color`: the color to convert. Can be a list, numpy array, or tuple – e.g. `(255, 0, 0)`
  - `code`: an OpenCV color conversion code, e.g. `cv2.COLOR_HSV2BGR`
- Color constants (in BGR): `MAROON`, `RED`, `ORANGE`, `YELLOW`, `OLIVE`, `GREEN`, `PURPLE`, `FUCHSIA`, `LIME`, `TEAL`, `AQUA`, `BLUE`,
`NAVY`, `BLACK`, `GRAY`, `SILVER`, `WHITE`.

### `contours.py`

- `sort_contour(contour)`: sorts a contour's points in clockwise order.
  - `contour`: a numpy array of points with shape `(n_points, 1, 2)`
- `center(contour)`: returns the center of a contour as tuple `(cx, cy)`
- `iou(contour1, contour2)`: returns the intersection over union (IOU) of two contours. IOU = (area of intersection) / (area of union)
