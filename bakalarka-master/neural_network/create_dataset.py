import cv2
import os
from image_preprocess import utilities


def save_cell_image(img_warped, x1, y1, x2, y2, save_dir='neural_network/data/images'):
    """
    Saves a cropped grayscale image of a specified cell region from the warped table image.

    The function creates the save directory if it does not exist, determines the next available
    filename (with a three-digit number), crops the cell region, converts it to grayscale, and saves it as a PNG file.

    Parameters:
        img_warped (np.ndarray): Warped (aligned) image of the table.
        x1 (int): Left coordinate of the cell bounding box.
        y1 (int): Top coordinate of the cell bounding box.
        x2 (int): Right coordinate of the cell bounding box.
        y2 (int): Bottom coordinate of the cell bounding box.
        save_dir (str, optional): Directory where the cropped cell image will be saved.

    Returns:
        None. The cropped cell image is saved to disk.
    """

    os.makedirs(save_dir, exist_ok=True)

    # Find the next available filename
    existing = [f for f in os.listdir(save_dir) if f.endswith('.png') and f[:3].isdigit()]
    if existing:
        max_num = max(int(f[:3]) for f in existing)
        next_num = max_num + 1
    else:
        next_num = 0

    filename = f"{next_num:03d}.png"
    filepath = os.path.join(save_dir, filename)

    cell_img = img_warped[y1:y2, x1:x2]

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(filepath, gray)


def cut_and_save_cells(img_warped, centers, indexes, cell_w, cell_h):
    """
    Iterates over specified cell indexes, crops each cell from the warped table image, and saves it as a grayscale PNG.

    For each index, calculates the bounding box using the cell centers and size, prints the coordinates,
    and calls save_cell_image to save the cropped cell.

    Parameters:
        img_warped (np.ndarray): Warped (aligned) image of the table.
        centers (list[list[tuple[int, int]]]): 2D list of cell center coordinates (x, y).
        indexes (list[tuple[int, int]]): List of (row, column) indexes specifying which cells to save.
        cell_w (float): Width of each cell.
        cell_h (float): Height of each cell.

    Returns:
        None. Cropped cell images are saved to disk.
    """

    for index in indexes:
        x1, y1, x2, y2 = utilities.cut_cell(index, centers, cell_w, cell_h)

        print(f"Saving cell at: ({x1}, {y1}), ({x2}, {y2})")

        save_cell_image(img_warped, x1, y1, x2, y2)

