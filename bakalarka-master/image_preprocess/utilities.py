from image_preprocess import CONSTANTS


def cut_cell(index, centers, cell_w, cell_h):
    """
        Calculates the bounding box coordinates for a cell in the table based on its center and cell size.

        Parameters:
            index (tuple[int, int]): Tuple (row, column) specifying the cell's position.
            centers (list[list[tuple[int, int]]]): 2D list of cell center coordinates (x, y).
            cell_w (float): Width of the cell.
            cell_h (float): Height of the cell.

        Returns:
            tuple[int, int, int, int]: Coordinates (x1, y1, x2, y2) defining the bounding box of the cell.
    """

    index_x, index_y = index
    cx, cy = centers[index_x][index_y]

    # need to keep the +1 to avoid cutting issues for CNN
    x1 = int(cx - cell_w / CONSTANTS.CELL_MIN_THRESHOLD_HEIGHT + 1)
    y1 = int(cy - cell_h / CONSTANTS.CELL_MIN_THRESHOLD_WIDTH)
    x2 = int(cx + cell_w / CONSTANTS.CELL_MIN_THRESHOLD_HEIGHT)
    y2 = int(cy + cell_h / CONSTANTS.CELL_MIN_THRESHOLD_WIDTH)

    return x1, y1, x2, y2
