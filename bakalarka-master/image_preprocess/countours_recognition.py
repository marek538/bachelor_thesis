import cv2

from image_preprocess import CONSTANTS
from image_preprocess import IO_image


def find_contours(img):
    """
    Detects contours in the input image using Canny edge detection and returns the list of contours.

    Parameters:
        img (np.ndarray): Input image in BGR format.

    Returns:
        list[np.ndarray]: List of detected contours, where each contour is a NumPy array of points.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, CONSTANTS.CONTOUR_BLUR_KERNEL_SIZE, CONSTANTS.CONTOUR_BLUR_SIGMAX)
    edges = cv2.Canny(blur, CONSTANTS.CONTOUR_CANNY_THRESHOLD1,
                      CONSTANTS.CONTOUR_CANNY_THRESHOLD2)  # <class 'numpy.ndarray'>

    # IO_image.show_image(blur)

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, CONSTANTS.COLOR_GREEN, 2)
    IO_image.show_image(img_contours)
    return contours
