import json


def load_reference_answers(filepath):
    """
    Loads the JSON file containing correct answers and returns a dictionary mapping test versions
    to answer keys and their respective points.

    Parameters:
        filepath (str): Path to the JSON file with reference answers.

    Returns:
        dict[int, dict[int, dict[str, any]]]: Dictionary where each key is a test version (int)
                                              and each value is a dictionary mapping question numbers (int)
                                              to a dictionary containing "answer" (str) and "points" (int).

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)

        reference_dict = {}

        for version_str, answers in data.items():
            # Convert the test version to an integer (e.g., "14" -> 14)
            version_int = int(version_str)

            # Convert the question numbers to integers and keep the nested dictionary (answer and points)
            formatted_answers = {int(q_num): ans_data for q_num, ans_data in answers.items()}

            reference_dict[version_int] = formatted_answers

        return reference_dict
    except Exception as e:
        raise RuntimeError(f"Error reading reference answers: {e}")


def extract_answers_from_results(results_filepath):
    """
    Loads the scan results JSON and extracts student IDs, answer indexes, and review flags for each file.

    Parameters:
        results_filepath (str): Path to the JSON file with scan results.

    Returns:
        dict[str, dict]: A dictionary where each key is a filename (str) and each value is a dictionary with:
                         - 'student_id' (str or None): The student ID extracted from the QR code.
                         - 'indexes_of_crosses' (list): List of (row, column) tuples representing detected answers.
                         - 'needs_manual_review' (bool): Indicates if the scan requires manual review.
                         - 'score_box_coordinates' (dict): Coordinates for score annotations.
                         - 'output_file_name' (str): Name of the output file.
                         - 'filled_cells' (list): List of (row, column) tuples representing all filled cells.

    Raises:
        RuntimeError: If the file cannot be read or parsed.
    """
    try:
        with open(results_filepath, "r", encoding="utf-8") as file:
            scan_results = json.load(file)

        extracted_data = {}

        for filename, data in scan_results.items():
            student_id = data.get("student_id")
            final_answers = data.get("final_answers", [])
            filled_cells = data.get("full_cells", [])
            needs_manual_review = data.get("needs_manual_review", False)
            score_box_coordinates = data.get("score_box_coordinates", {})
            output_file_name = data.get("output_file_name", "")

            extracted_data[filename] = {
                "student_id": student_id,
                "indexes_of_crosses": final_answers,
                "needs_manual_review": needs_manual_review,
                "score_box_coordinates": score_box_coordinates,
                "output_file_name": output_file_name,
                "filled_cells": filled_cells
            }

        return extracted_data
    except Exception as e:
        raise RuntimeError(f"Error reading scan results: {e}")
