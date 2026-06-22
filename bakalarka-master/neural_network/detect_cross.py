import torch
from torchvision import transforms
from PIL import Image
from neural_network.training_network import TRAINING_CONSTANTS


def predict_image(img, model, confidence_threshold=0.71):
    """
    Predicts whether a cell image is filled (crossed) using a trained neural network model.

    The function preprocesses the input image to match the model's expected input, performs inference,
    and returns the predicted class. If the model's confidence is below the specified threshold,
    it returns class 2 to indicate uncertainty.

    Parameters:
        img (np.ndarray): Input cell image as a NumPy array.
        model (torch.nn.Module): Trained PyTorch model for classification.
        confidence_threshold (float, optional): Minimum probability required to accept the prediction.
                                               If the maximum probability is below this value, returns 2 (uncertain).

    Returns:
        int: Predicted class label:
             - 0: Cell is empty (not crossed)
             - 1: Cell is filled (crossed)
             - 2: Uncertain (confidence below threshold)
    """

    # 1. Convert the input array to a PIL Image
    img = Image.fromarray(img)

    # 2. Apply validation transforms (must match exactly)
    predict_transforms = transforms.Compose([
        transforms.Resize((TRAINING_CONSTANTS.CELL_WIDTH, TRAINING_CONSTANTS.CELL_HEIGHT)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3. Transform the image and add a batch dimension (B, C, H, W)
    img_tensor = predict_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)

    # 4. Move the tensor to the correct device (CPU or GPU)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        # 5. Get the raw model output
        output = model(img_tensor)

        # 6. Convert log probabilities to standard probabilities (0.0 to 1.0)
        probabilities = torch.exp(output)

        # 7. Find the highest probability and the corresponding class (0 or 1)
        max_prob, predicted = torch.max(probabilities, 1)

        # Extract the python numbers from the PyTorch tensors
        max_prob_value = max_prob.item()
        predicted_class = predicted.item()

        # 8. THE REJECTION OPTION
        # If the model's confidence is too low, return class '2' (Uncertain)
        if max_prob_value < confidence_threshold:
            return 2
        else:
            return predicted_class
