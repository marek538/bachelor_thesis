import torch

# Adjust the imports according to your project structure
from neural_network.training_network.train_init import initialize_training
from neural_network.load_network import load_model


def print_probabilities_for_latex(model_path):
    """
        Loads a trained model and validation data, computes the predicted probability for class 1 for each sample,
        and prints the true label and probability in a format suitable for LaTeX/pgfplots.

        Parameters:
            model_path (str): Path to the saved PyTorch model (.pth file).

        Returns:
            None. Results are printed to the console in CSV format.
    """

    # 1. Load data and device
    device, _, val_loader, _ = initialize_training()

    # 2. Load the model
    model = load_model(model_path)
    if model is None:
        print("Failed to load the model.")
        return

    model = model.to(device)
    model.eval()

    print("--- COPY THE DATA BELOW THIS LINE ---")

    # Print the header exactly as pgfplots expects it
    print("true_label,probability")

    # 3. Iterate through validation data
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.exp(outputs)

            # Extract probability ONLY for Class 1 (the 2nd column)
            prob_class_1 = probabilities[:, 1].cpu().numpy()
            true_labels = labels.cpu().numpy()

            # Print each result directly to the console
            for i in range(len(true_labels)):
                # Format: label, probability (rounded to 6 decimal places)
                print(f"{int(true_labels[i])},{float(prob_class_1[i]):.6f}")

    print("--- STOP COPYING HERE ---")


if __name__ == "__main__":
    # Point this to your saved .pth file
    model_path = "../data/models/dynamic_cnn_best_new_model.pth"
    print_probabilities_for_latex(model_path)