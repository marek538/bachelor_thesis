import cv2


def draw_legend_and_score(img, score, second_to_last_cell, last_cell):
    """
    Draws a legend and the score on the given image.

    Parameters:
        img (numpy.ndarray): The image on which the legend and score will be drawn.
        score (int): The score to display in the legend.
        second_to_last_cell (tuple[int, int]): Coordinates (x, y) of the second-to-last cell.
        last_cell (tuple[int, int]): Coordinates (x, y) of the last cell.

    Returns:
        numpy.ndarray: The image with the legend and score drawn on it.
    """

    x1, y1 = count_legend_coordinates(second_to_last_cell, last_cell)

    img_height, img_width = img.shape[:2]
    scale_factor = img_width / 2000.0
    current_x = int(x1)
    y_offset = int(y1)

    parts = [
        (f"Score: {score} ", 1.2 * scale_factor, (0, 0, 0), max(1, int(3 * scale_factor))),
        ("Blue crosses: Accepted as answer ", 0.8 * scale_factor, (255, 0, 0), max(1, int(2 * scale_factor))),
        ("Red crosses: Rejected due to invalid mark ", 0.8 * scale_factor, (0, 0, 255), max(1, int(2 * scale_factor))),
        ("Points in rightmost cell.", 0.8 * scale_factor, (0, 0, 0), max(1, int(2 * scale_factor)))
    ]

    for text, font_scale, color, thickness in parts:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(img, text, (current_x, y_offset), font, font_scale, color, thickness)
        current_x += text_width
    return img


def count_legend_coordinates(second_to_last_cell, last_cell):
    """
    Calculates the coordinates for placing the legend based on the positions of the second-to-last
    and last cells in the score box.

    Parameters:
        second_to_last_cell (tuple[int, int]): Coordinates (x, y) of the second-to-last cell.
        last_cell (tuple[int, int]): Coordinates (x, y) of the last cell.

    Returns:
        tuple[float, float]: Calculated coordinates (x, y) for the legend placement.
    """

    x1, y1 = second_to_last_cell
    x2, y2 = last_cell

    dx = x2 - x1
    dy = y2 - y1

    p3_x = x2 + 2 * dx
    p3_y = y2 + 2 * dy

    final_x = p3_x / 8
    final_y = p3_y

    return final_x, final_y

