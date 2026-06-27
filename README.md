# OMR System for University Entrance Exams 🎓

An automated Optical Mark Recognition (OMR) pipeline developed for processing large-scale multiple-choice answer sheets. Designed to drastically reduce manual grading time and eliminate human error during university admission exams.

## 🚀 Overview
This project digitizes and evaluates scanned answer sheets using a hybrid approach of classic Computer Vision and Deep Learning. Instead of relying purely on rigid pixel-counting heuristics, it employs a custom Convolutional Neural Network (CNN) to robustly classify student answers, seamlessly handling messy handwriting, corrections, and cross-outs.

## ⚙️ The Pipeline
1. **Localization & Alignment:** Uses **ArUco markers** and projective geometry to precisely locate and align the answer grid, correcting any rotation or skew introduced by the scanner.
2. **Cell Extraction:** Calculates coordinates and extracts individual answer bubbles from the document.
3. **CNN Classification:** A custom-trained **PyTorch** model evaluates each cell to determine the student's intent.
4. **Human-in-the-Loop:** Incorporates a strict confidence threshold. If the network's confidence drops below a set threshold (e.g., due to highly ambiguous scribbles), the specific cell is flagged for manual review.
5. **Data Export:** Decodes the applicant's QR code and aggregates the final scores into a structured `CSV`/`JSON` format.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, NumPy
* **Deep Learning:** PyTorch
* **Optimization:** Optuna (Hyperparameter tuning)

