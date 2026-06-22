import argparse
import csv
import os
import cv2
from image_preprocess import IO_image, CONSTANTS
from evaluate_score import get_json_data, safety_features, draw_legend_and_score


def parse_sheet(indexes_of_crosses):
    """
    Parses the list of marked cell indexes to extract the test version and the student's answers.

    Parameters:
        indexes_of_crosses (list of tuple): List of (row, column) indexes where crosses were detected.

    Returns:
        version (int): The test version number extracted from the first row.
        results (dict[int, str]): Dictionary mapping exercise numbers to selected answer letters.
    """

    iterator = 0
    version = 0

    # version: located in first row
    second_half = False
    while True:
        if indexes_of_crosses[iterator][0] == 1:
            if not second_half and indexes_of_crosses[iterator][1] >= 5:
                second_half = True
                version = 0
            version = version * 10 + ((indexes_of_crosses[iterator][1] + 1) % 5)
        else:
            break

        iterator += 1

    # get all answers for each exercise
    lines: dict[int, list[int]] = {}
    for i in range(iterator, len(indexes_of_crosses)):
        exercise = indexes_of_crosses[i][0] - 3  # padding of 3 fist rows being extra and 1 due to array indexing
        if exercise not in lines:
            lines[exercise] = []
        lines[exercise].append(indexes_of_crosses[i][1])

    results: dict[int, str] = {}
    num_to_char = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

    # parse correct answer in each line
    for key, value_list in lines.items():
        # if there are more than 1 cross, count only those in second half
        if len(value_list) > 1:
            tmp = [v for v in value_list if v >= 5]
            value_list = tmp

        # half are separated, correct if only one cross remains, otherwise, don't add anything to results
        if len(value_list) == 1:
            results[key] = num_to_char[value_list[0] % 5]

    return version, results


def main(scan_results, reference_answers, output_csv):
    """
        Compares recognized answers from scanned forms with reference answers and writes the scores to a CSV file.

        Parameters:
            scan_results (str): Path to the JSON file with scan results.
            reference_answers (str): Path to the JSON file with reference answers.
            output_csv (str, optional): Path to the output CSV file for final scores.

        Returns:
            None. Results are saved to the specified CSV file.
    """
    try:
        data_answers = get_json_data.extract_answers_from_results(scan_results)
        data_reference = get_json_data.load_reference_answers(reference_answers)

        # Initialize a list to store the rows for the CSV file
        # The first item is the header row
        csv_rows = [["Filename", "Student ID", "Score", "error_message"]]

        for filename, student_data in data_answers.items():
            print("Evaluating file:", filename)

            directory = os.path.dirname(scan_results)
            current_image_path = directory + "/" + student_data.get("output_file_name", "")
            current_image = IO_image.read_image(current_image_path)

            img_height, img_width = current_image.shape[:2]
            scale_factor = img_width / 2000.0
            dynamic_font_scale = 1.0 * scale_factor
            dynamic_thickness = max(1, int(5 * scale_factor))

            # Retrieve the student ID, default to an empty string if missing or None
            student_id = student_data.get("student_id") or ""
            indexes_of_crosses = student_data.get("indexes_of_crosses", [])

            if not indexes_of_crosses:
                csv_rows.append([filename, student_id, 0, "No crosses"])
                continue

            if student_id == "":
                csv_rows.append([filename, student_id, 0, "QR code not read"])
                continue

            if student_data.get("needs_manual_review", True):
                csv_rows.append([filename, student_id, 0, "cross misclassified, needs manual review"])
                continue

            version, student_answers = parse_sheet(indexes_of_crosses)

            if version in data_reference:
                correct_answers = data_reference[version]
                score = 0

                # Calculate the score
                for question_num, answer_data in correct_answers.items():

                    correct_ans = answer_data["answer"]
                    points = answer_data["points"]

                    if question_num in student_answers and student_answers[question_num] == correct_ans:
                        score += points

                        # + 3 constant because in cell centers first 3 rows are reserved for version and padding
                        cv2.putText(current_image, str(points),
                                    student_data["score_box_coordinates"].get(str(question_num + 3), (0, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, dynamic_font_scale,
                                    CONSTANTS.COLOR_GREEN, dynamic_thickness)
                        # IO_image.show_image(current_image)

                # Append the successful result to our rows list

                if safety_features.empty_row(student_data["indexes_of_crosses"], student_data["filled_cells"]):
                    csv_rows.append([filename, student_id, score,
                                     "needs manual review, empty row detected where an object was discarded"])
                else:
                    csv_rows.append([filename, student_id, score, ""])

                # write legend on the image and save it
                current_image = (draw_legend_and_score.
                                 draw_legend_and_score(current_image, score,
                                                       student_data["score_box_coordinates"].get("18"),
                                                       student_data["score_box_coordinates"].get("19")))

                # new_image_path = directory + "/" + "graded_" + student_data.get("output_file_name", "")
                cv2.imwrite(current_image_path, current_image)

            else:
                # Append an error message to the score column if the version is missing
                csv_rows.append([filename, student_id, 0, f"Error: Version {version} not found"])

        # Write all gathered rows to the CSV file
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)

        print(f"Results successfully saved to {output_csv}")
        return

    except Exception as e:
        print(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scanned forms and compare two JSON files.")

    # Argument for the first JSON file
    parser.add_argument(
        "-r1", "--results",
        type=str,
        required=False,
        default="./results/scan_results.json",
        help="Path to the scan_results JSON file."
    )

    # Argument for the second JSON file
    parser.add_argument(
        "-r2", "--reference",
        type=str,
        required=False,
        default="./reference_answers.json",
        help="Path to the reference_answers JSON file."
    )

    parser.add_argument(
        "-o", "--output_csv",
        type=str,
        required=False,
        default="./results/final_scores.csv",
        help="Path to the output CSV file for final scores."
    )

    args = parser.parse_args()

    # Call the main function with the parsed JSON files
    main(scan_results=args.results, reference_answers=args.reference, output_csv=args.output_csv)
