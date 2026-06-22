import numpy as np
import cv2
from image_preprocess import CONSTANTS


def inverse_matrix(matrix):
    """
        Computes the inverse of a given transformation matrix and determines its type.

        Parameters:
            matrix (np.ndarray): The transformation matrix to be inverted.
                                 It can be either a 3x3 matrix (perspective transformation)
                                 or a 2x3 matrix (affine transformation).

        Returns:
            tuple: A tuple containing:
                - inv_matrix (np.ndarray): The inverted transformation matrix.
                - transform_type (str): The type of transformation ("perspective" or "affine").

        Raises:
            ValueError: If the input matrix shape is not (3, 3) or (2, 3).
    """
    # it is not enough to return inv_matrix
    if matrix.shape == (3, 3):
        # 4 points: Perspective Transformation
        _, inv_matrix = cv2.invert(matrix)
        transform_type = "perspective"
    elif matrix.shape == (2, 3):
        # 3 points: Affine Transformation
        inv_matrix = cv2.invertAffineTransform(matrix)
        transform_type = "affine"
    else:
        raise ValueError("Unexpected transformation matrix shape: {}".format(matrix.shape))

    return inv_matrix, transform_type


def get_original_coordinates(point_warped, inv_matrix, transform_type):
    """
        Maps a point from the warped (transformed) space back to the original space
        using the inverse transformation matrix.

        Parameters:
            point_warped (np.ndarray): The coordinates of the point in the warped space,
                                       provided as a NumPy array of shape (1, 1, 2).
            inv_matrix (np.ndarray): The inverse transformation matrix.
            transform_type (str): The type of transformation ("perspective" or "affine").

        Returns:
            tuple: A tuple containing:
                - orig_cx (int): The x-coordinate of the point in the original space.
                - orig_cy (int): The y-coordinate of the point in the original space.
    """
    if transform_type == "perspective":
        point_original = cv2.perspectiveTransform(point_warped, inv_matrix)
    else:
        # Affine transformation uses cv2.transform instead of perspectiveTransform
        point_original = cv2.transform(point_warped, inv_matrix)

    # 4. Extract the original X and Y coordinates and convert them to integers
    orig_cx = int(point_original[0][0][0])
    orig_cy = int(point_original[0][0][1])
    return orig_cx, orig_cy


def warp_table(img, aruco_positions, output_size=(CONSTANTS.WARPED_IMAGE_WIDTH, CONSTANTS.WARPED_IMAGE_HEIGHT)):
    """
        Warps the input image so that the table is aligned based on detected ArUco marker positions.

        Parameters:
            img (np.ndarray): The input image containing the table and ArUco markers.
            aruco_positions (dict): Dictionary mapping marker IDs (int) to their detected corner coordinates.
            output_size (tuple, optional): Desired size (width, height) of the output warped image.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - The warped (aligned) image as a NumPy array.
                - The transformation matrix used for warping.

        Raises:
            ValueError: If fewer than 3 ArUco markers are detected.
    """

    # fails if not at least 3 markers found
    if len(aruco_positions) < 3:
        raise ValueError("Not enough ArUco markers detected to perform warping.")

    table_edges = {}
    for marker_id, pts in aruco_positions.items():
        # print(type(pts))
        if marker_id == 1:
            # left up coordinate
            table_edges[marker_id] = pts[0]
        elif marker_id == 2:
            # right up coordinate
            table_edges[marker_id] = pts[1]
        elif marker_id == 3:
            # left down coordinate
            table_edges[marker_id] = pts[0]
        elif marker_id == 4:
            # right down coordinate
            table_edges[marker_id] = pts[1]

    # size of the output image
    w, h = output_size
    dst_points = {
        1: [0, 0],  # left up
        2: [w - 1, 0],  # right up
        3: [0, h - 1],  # left down
        4: [w - 1, h - 1]  # right down
    }

    # array of corresponding points
    src = np.array([table_edges[i] for i in table_edges.keys()], dtype=np.float32)
    dst = np.array([dst_points[i] for i in table_edges.keys()], dtype=np.float32)

    if src.shape[0] == 4:
        print("4 markers found, using perspective transformation")
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, matrix, (w, h))
    else:
        print("3 markers found, using affine transformation")
        matrix = cv2.getAffineTransform(src[:3], dst[:3])
        warped = cv2.warpAffine(img, matrix, (w, h))

    return warped, matrix
