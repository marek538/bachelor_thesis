import cv2
from image_preprocess import CONSTANTS


def map_table_cells(img_warped):
    """
    Calculates the centers and size of answer cells in the warped table image.

    The table layout consists of one longer cell for ID, 10 answer cells, and a grade cell per row.
    There are 20 rows in total, with some rows (e.g., 1, 3, 4) being descriptive and skipped.

    Parameters:
        img_warped (np.ndarray): Warped (aligned) image of the table.

    Returns:
        list: A list containing:
            - centers (list[list[tuple[int, int]]]): 2D list of cell center coordinates (x, y) for each row and column.
            - cell_w (float): Width of each answer cell.
            - cell_h (float): Height of each answer cell.
            - answer_centers (dict): Dictionary mapping row indices to the center coordinates of the grade cell.
    """

    height, width = img_warped.shape[:2]
    cell_w = (width - CONSTANTS.TABLE_FIRST_COL_WIDTH - CONSTANTS.TABLE_LAST_COL_WIDTH) / CONSTANTS.TABLE_COLS
    cell_h = height / CONSTANTS.TABLE_ROWS

    centers = []
    img_warped_to_show = img_warped.copy()

    answer_centers = {}

    for r in range(CONSTANTS.TABLE_ROWS):
        centers.append([])
        for c in range(CONSTANTS.TABLE_COLS + 1):
            if r in CONSTANTS.TABLE_SKIPPED_ROWS:
                continue
            cx = int(CONSTANTS.TABLE_FIRST_COL_WIDTH + c * cell_w + cell_w / 2)
            cy = int(r * cell_h + cell_h / 2)

            if c == CONSTANTS.TABLE_COLS:
                answer_centers[r] = (cx, cy)
                continue

            centers[len(centers) - 1].append((cx, cy))
            cv2.circle(img_warped_to_show, (cx, cy), 5, CONSTANTS.COLOR_GREEN, -1)

    # IO_image.show_image(img_warped_to_show)

    return [centers, cell_w, cell_h, answer_centers]
