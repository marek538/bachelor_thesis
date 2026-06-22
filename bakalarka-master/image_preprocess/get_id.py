from pyzbar.pyzbar import decode
import cv2
from image_preprocess.table_finder import no_preprocess, preprocess_median_filter, preprocess_gaussian_filter
from image_preprocess import CONSTANTS


def get_student_id_from_qr_code(img):
    """
    Detects and decodes a QR code from the input image to extract the student ID.

    Parameters:
        img (np.ndarray): Input image in which to search for a QR code.

    Returns:
        tuple[str | None, np.ndarray]: A tuple containing:
            - The decoded student ID as a string if a QR code is found, otherwise None.
            - The (possibly cropped or annotated) image as a NumPy array.

    Raises:
        None. If no QR code is found, returns (None, original image).
    """

    preprocess_functions = [no_preprocess, preprocess_median_filter, preprocess_gaussian_filter]

    qr_codes = []

    for function in preprocess_functions:
        tmp = function(img)
        qr_codes = decode(tmp)
        if len(qr_codes) > 0:
            (x, y, w, h) = qr_codes[0].rect
            cv2.rectangle(img, (x, y), (x + w, y + h), CONSTANTS.COLOR_GREEN, 3)
            cv2.putText(img, qr_codes[0].data.decode("utf-8"), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, CONSTANTS.COLOR_GREEN, 2)
            break

    if len(qr_codes) == 0:
        print("No QR code detected in the image.")
        return None, img

    # IO_image.show_image(img)
    print(f"QR code detected with data: {qr_codes[0].data.decode('utf-8')} in image")

    return qr_codes[0].data.decode("utf-8"), img

