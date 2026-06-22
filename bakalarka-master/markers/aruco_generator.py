import cv2
import cv2.aruco as aruco
from image_preprocess import CONSTANTS


def generate_aruco_markers():
    """
    Generates ArUco marker images with IDs 1 to 4 using a predefined dictionary and saves them as PNG files.

    The marker size is determined by CONSTANTS.GENERATED_MARKER_SIZE. Generated images are saved as
    'marker_1.png', 'marker_2.png', etc., in the current directory.

    Returns:
        None. Marker images are saved to disk.
    """

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    marker_size = CONSTANTS.GENERATED_MARKER_SIZE
    font = cv2.FONT_HERSHEY_SIMPLEX

    for marker_id in range(1, 5):
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        cv2.imwrite(f"marker_{marker_id}.png", marker_img)

    print("Markers generated and saved in 'markers' directory.")


if __name__ == "__main__":
    generate_aruco_markers()
