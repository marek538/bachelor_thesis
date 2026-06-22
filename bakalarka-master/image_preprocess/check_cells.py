import cv2
from image_preprocess import CONSTANTS
from image_preprocess import utilities


def check_cells(img_warped, centers, cell_w, cell_h):
    """
    Checks each cell in the warped table image to determine which cells are filled with black pixels and require
    further analysis.

    Parameters:
        img_warped (np.ndarray): Warped (aligned) image of the table.
        centers (list[list[tuple[int, int]]]): 2D list of cell center coordinates (x, y).
        cell_w (int): Width of each cell.
        cell_h (int): Height of each cell.

    Returns:
        list[list[int, int]]: List of [row, column] indexes for cells detected as filled.
    """

    print("Checking cells for filled content...")
    img = img_warped.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = []

    for index_x in range(len(centers)):

        for index_y in range(len(centers[index_x])):
            cx, cy = centers[index_x][index_y]

            x1, y1, x2, y2 = utilities.cut_cell((index_x, index_y), centers, cell_w, cell_h)

            cell_roi = gray[y1:y2, x1:x2]

            _, thresh = cv2.threshold(cell_roi, CONSTANTS.CELL_THRESHOLD_BINARY,
                                      255, cv2.THRESH_BINARY_INV)
            non_zero_count = cv2.countNonZero(thresh)

            if non_zero_count > CONSTANTS.THRESHOLD_MINIMAL_POINTS:
                result.append([index_x, index_y])
                cv2.circle(img, (cx, cy), 10, CONSTANTS.COLOR_RED, -1)

    # IO_image.show_image(img)
    return result
