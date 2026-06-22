import cv2

from image_preprocess import CONSTANTS


def preprocess_gaussian_filter(img):
    """
    Converts the input image to grayscale and applies a Gaussian blur followed by adaptive thresholding.

    Parameters:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Preprocessed image suitable for ArUco marker detection.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, CONSTANTS.ARUCO_BLUR_KERNEL_SIZE, CONSTANTS.ARUCO_BLUR_SIGMAX)
    # gray = cv2.medianBlur(gray, 5)
    # _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # !!! this works great for rotated inputs, otherwise it can mess up the detection
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25, 10
    )

    return gray


def preprocess_median_filter(img):
    """
    Converts the input image to grayscale and applies a median blur.

    Parameters:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Preprocessed grayscale image.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    return gray


def no_preprocess(img):
    return img


def find_aruco(img):
    """
        Detects ArUco markers in the input image using multiple preprocessing strategies.

        Parameters:
            img (np.ndarray): Input image in BGR format.

        Returns:
            tuple: (corners, ids)
                - corners (list of np.ndarray)
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    preprocess_functions = [no_preprocess, preprocess_median_filter, preprocess_gaussian_filter]

    for function in preprocess_functions:
        gray = function(img)
        # IO_image.show_image(gray)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) >= 3:
            return corners, ids

    raise Exception("No enough ArUco markers found in the image.")


def locate_table(img):
    """
    Locates the table in the input image by detecting ArUco markers and annotates their positions.

    Parameters:
        img (np.ndarray): Input image in BGR format.

    Returns:
        dict[int, np.ndarray]: Dictionary mapping marker IDs to their corner coordinates (4x2 array of int).
    """

    corners, ids = find_aruco(img)

    results = {}
    img_out = img.copy()

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            pts = corners[i].reshape((4, 2)).astype(int)
            results[int(marker_id)] = pts

            cv2.polylines(img_out, [pts], isClosed=True, color=CONSTANTS.COLOR_GREEN, thickness=2)

            x, y = pts[0]
            cv2.putText(img_out, f"ID {marker_id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONSTANTS.COLOR_RED, 2)

    # IO_image.show_image(img_out)

    return results  # dictionary - id : coordinates
