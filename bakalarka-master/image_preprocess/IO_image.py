import cv2
import matplotlib.pyplot as plt
import fitz
import numpy as np
import os


def read_pdf_as_images(name, dpi=200):
    """
    Reads a PDF file and returns a list of OpenCV-compatible images (NumPy arrays).
    """
    file_path = str(name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The PDF file was not found: {file_path}")

    try:
        document = fitz.open(str(name))
    except Exception as e:
        print(f"{name} failed to read. Error: {e}")
        return None

    images = []

    for page_number in range(len(document)):
        page = document.load_page(page_number)
        pixmap = page.get_pixmap(dpi=dpi)
        img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.h, pixmap.w, pixmap.n)

        if pixmap.n == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pixmap.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)

        images.append(img_array)

    document.close()
    return images


def read_image(name):
    img = cv2.imread(str(name))

    if img is None:
        print(f"{name} failed to read")
        return None

    return img


def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        plt.imshow(img_rgb)
        plt.title("")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Show plots in tool window is on - can't show image\n {e}")
