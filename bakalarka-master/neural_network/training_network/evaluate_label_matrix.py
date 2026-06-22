import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from neural_network.training_network.train_init import initialize_training
from neural_network.load_network import load_model


def evaluate_and_plot_matrix(model_path):
    """
    Loads a trained model and validation data, computes predictions, and displays the confusion matrix.

    Parameters:
        model_path (str): Path to the saved PyTorch model (.pth file).

    Returns:
        None. Prints confusion matrix statistics and displays a plot of the confusion matrix.
    """

    # 1. Load data and device
    device, _, val_loader, _ = initialize_training()

    # 2. Load your final tuned model
    model = load_model(model_path)
    if model is None:
        print("Failed to load the model. Please check the path.")
        return

    model = model.to(device)
    model.eval()  # Crucial: switches the model to evaluation/prediction mode

    all_predictions = []
    all_true_labels = []

    print("Running predictions on the validation dataset...")

    # 3. Iterate through the validation set and collect predictions
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Convert tensors back to plain numbers (numpy) and store them in the lists
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    # 4. Calculate the Confusion Matrix
    cm = confusion_matrix(all_true_labels, all_predictions)

    # Unpack the components (this specific unpacking works for binary classification - 2 classes)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 40)
    print("CONFUSION MATRIX RESULTS")
    print("=" * 40)
    print(f"True Negatives  (Correctly Class 0):   {tn}")
    print(f"False Positives (Incorrectly Class 1): {fp}  <-- False alarm (sees a cross where there isn't one)")
    print(f"False Negatives (Incorrectly Class 0): {fn}  <-- Missed (missed an actual cross)")
    print(f"True Positives  (Correctly Class 1):   {tp}")
    print("=" * 40)

    # 5. Plot the graph
    # (Class 0 = Empty cell, Class 1 = Crossed cell)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues)

    plt.show()


if __name__ == "__main__":
    # Path to your dynamic model
    model_path = "../data/models/dynamic_cnn_best_new_model.pth"
    evaluate_and_plot_matrix(model_path)
