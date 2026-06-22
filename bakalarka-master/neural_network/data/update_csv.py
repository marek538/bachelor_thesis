import csv
import os


def update_csv(create_new: bool = False, default='2'):
    """
    Updates or creates a CSV file with image filenames and their labels for training data.

    If create_new is True, a new CSV file is created with all PNG images in the 'images' directory,
    each assigned the default label. If create_new is False, the function updates the existing CSV file
    by adding any new images not already present, assigning them the default label.

    Parameters:
        create_new (bool, optional): If True, creates a new CSV file from scratch. If False, updates the existing file.
        default (str, optional): Default label to assign to new images.

    Returns:
        None. The CSV file 'labels.csv' is created or updated in the current directory.
    """

    images_dir = 'images'
    csv_path = 'labels.csv'

    all_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    all_images_sorted = sorted(
        all_images,
        key=lambda x: int(os.path.splitext(x)[0])
    )

    new_rows = []
    rows = []
    if create_new:
        print("Creating new CSV file.")
        for img in all_images_sorted:
            new_rows.append({'filename': img, 'label': default})

    else:
        print("Updating existing CSV file.")
        existing = set()

        if not create_new and os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing.add(row['filename'])
                    rows.append(row)

        for img in all_images_sorted:
            if img not in existing:
                new_rows.append({'filename': img, 'label': default})

    if create_new:
        rows = []

    rows.extend(new_rows)

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'label'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


update_csv(create_new=False, default='1')
