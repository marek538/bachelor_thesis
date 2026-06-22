def empty_row(filled_cells, final_answers):
    """
    Checks if there are any rows with filled cells that are missing in the final answers.

    Parameters:
        filled_cells (list[tuple[int, int]]): List of (row, column) tuples representing all filled cells.
        final_answers (list[tuple[int, int]]): List of (row, column) tuples representing the final detected answers.

    Returns:
        bool: True if there are rows with filled cells that are not in the final answers, False otherwise.
    """

    filled_rows = {cell[0] for cell in filled_cells if cell[0] > 1}
    final_rows = {cell[0] for cell in final_answers if cell[0] > 1}

    missing_rows = filled_rows - final_rows

    return len(missing_rows) > 0
