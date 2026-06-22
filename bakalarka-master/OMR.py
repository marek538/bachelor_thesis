from pathlib import Path
import os
import json
from image_preprocess import align_table, check_cells, CONSTANTS, IO_image, map_table_cells, table_finder, utilities
from neural_network import load_network
from neural_network import detect_cross
from image_preprocess.get_id import get_student_id_from_qr_code
import cv2
import numpy as np
import argparse
import random


def process_image(img, model, scan_data):
    """
        Processes a single image to extract student ID, detect filled cells, and predict answers.

        Parameters:
            img (np.ndarray): The input image to process.
            model (torch.nn.Module): The pre-trained neural network model for prediction.
            scan_data (dict): Dictionary to store scan results and metadata.
        Returns:
            json: Updated scan_data dictionary with extracted information.
            image: The processed image with detected answers marked.
    """

    qr_code, image = get_student_id_from_qr_code(img)
    if qr_code is None:
        scan_data["error_message"] = "QR code not read."
        scan_data["needs_manual_review"] = True
    else:
        scan_data["student_id"] = qr_code

    found_markers = table_finder.locate_table(img)
    img_warped, matrix = align_table.warp_table(img, found_markers)

    centers, cell_w, cell_h, answer_centers = map_table_cells.map_table_cells(img_warped)

    indexes_of_full_centers = check_cells.check_cells(img_warped, centers, cell_w, cell_h)
    scan_data["full_cells"] = indexes_of_full_centers

    inv_matrix, transform_type = align_table.inverse_matrix(matrix)

    print("Predicting answers for detected filled cells...")
    for index in indexes_of_full_centers:
        x1, y1, x2, y2 = utilities.cut_cell(index, centers, cell_w, cell_h)
        cell_img = img_warped[y1:y2, x1:x2]
        prediction = detect_cross.predict_image(cell_img, model)

        cx, cy = centers[index[0]][index[1]]

        point_warped = np.array([[[cx, cy]]], dtype=np.float32)

        orig_cx, orig_cy = align_table.get_original_coordinates(point_warped, inv_matrix, transform_type)

        if prediction == 1:
            cv2.circle(img_warped, (cx, cy), 5, CONSTANTS.COLOR_BLUE, -1)
            cv2.circle(image, (orig_cx, orig_cy), 7, CONSTANTS.COLOR_BLUE, -1)
            scan_data["final_answers"].append(index)
        elif prediction == 0:
            cv2.circle(img_warped, (cx, cy), 5, CONSTANTS.COLOR_RED, -1)
            cv2.circle(image, (orig_cx, orig_cy), 7, CONSTANTS.COLOR_RED, -1)
        else:
            scan_data["error_message"] = f"Cell not recognized"
            scan_data["needs_manual_review"] = True
            cv2.circle(img_warped, (cx, cy), 5, CONSTANTS.COLOR_GREEN, -1)
            cv2.circle(image, (orig_cx, orig_cy), 7, CONSTANTS.COLOR_GREEN, -1)

    # IO_image.show_image(image)

    final_answer_centers = {}
    for idx, center in answer_centers.items():
        point_warped = np.array([[[center[0], center[1]]]], dtype=np.float32)
        orig_cx, orig_cy = align_table.get_original_coordinates(point_warped, inv_matrix, transform_type)
        final_answer_centers[idx] = (orig_cx, orig_cy)

    scan_data["score_box_coordinates"] = final_answer_centers
    return scan_data, image


def main(file_name, results_dir, prefix, dpi, model_path):
    """
    Main function to process a PDF file containing scanned forms. Processes each page of the PDF, extracts student IDs,
    detects filled cells, and saves the results.

    Parameters:
        file_name (str): Path to the input PDF file.
        results_dir (str): Directory to save the results.
        prefix (str): Prefix for the output image filenames.
        dpi (int): DPI for reading the PDF as images.
        model_path (str): Path to the pre-trained model for detecting crosses.
    """
    try:
        images = IO_image.read_pdf_as_images(file_name, dpi)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # root_directory = "neural_network/data/scans_for_training"
    # model = load_network.load_model("neural_network/data/models/dynamic_cnn.pth")
    model = load_network.load_model(model_path)

    results = {}

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(images)):
        print(f"Processing page {i + 1} of {len(images)} in file {os.path.basename(file_name)}...")

        img = images[i]

        scan_data = {
            "student_id": None,
            "needs_manual_review": False,
            "error_message": None,
            "full_cells": [],
            "final_answers": [],
            "score_box_coordinates": {},
            "output_file_name": None
        }

        try:
            scan_data, image = process_image(img, model, scan_data)

            if scan_data["student_id"] is None:
                output_file_name = f"{prefix}_NA_{random.randint(0, 0xffffffff):08x}.png"
            elif False:
                # only debugging because all the testing pdfs have the same student id
                output_file_name = output_file_name = f"{prefix}_{i + 1}.png"
            else:
                output_file_name = f"{prefix}_{scan_data['student_id']}.png"

            scan_data["output_file_name"] = output_file_name
            # Combine the directory path with the new filename
            result_filepath = os.path.join(results_dir, output_file_name)

            # Write the image to disk
            cv2.imwrite(result_filepath, image)
            results[output_file_name] = scan_data

        except Exception as e:
            print(os.path.basename(file_name), e)
            scan_data["error_message"] = str(e)
            scan_data["needs_manual_review"] = True

    with open(os.path.join(results_dir, "scan_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scanned forms and recognize cross marks.")
    parser.add_argument(
        "-f", "--file_name",
        type=str,
        default="scans_to_recognize/version_1.pdf",
        help="Path to the pdf containing scans."
    )
    parser.add_argument(
        "-r", "--results_dir",
        type=str,
        default="results",
        help="Directory where the results will be saved."
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="scan",
        help="Prefix to be used for the output files."
    )
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=200,
        help="DPI used for scanning pdf images"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        default="neural_network/data/models/dynamic_cnn.pth",
        help="Path to the pre-trained model for detecting crosses."
    )

    args = parser.parse_args()

    main(file_name=args.file_name, results_dir=args.results_dir, prefix=args.prefix, dpi=args.dpi,
         model_path=args.model_path)
