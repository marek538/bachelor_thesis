import optuna
import torch
import torch.nn as nn
from torch import optim
import os

from train_init import initialize_training
from early_stopper import EarlyStopping
from network_definition import DynamicConvNN
import TRAINING_CONSTANTS


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    Suggests hyperparameters for the convolutional neural network, builds and trains the model,
    evaluates its accuracy on the validation set, and returns the best validation accuracy achieved.
    Supports early stopping and Optuna pruning.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object for suggesting hyperparameters.

    Returns:
        float: Best validation accuracy (%) achieved during training.
    """

    device, train_loader, val_loader, train_dataset = initialize_training()

    # convolutional hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", TRAINING_CONSTANTS.MIN_CONV_LAYERS,
                                        TRAINING_CONSTANTS.MAX_CONV_LAYERS)
    out_channels_list = []
    kernel_sizes = []

    for i in range(num_conv_layers):
        out_channels_list.append(trial.suggest_categorical(f"out_channels_l{i}",
                                                           TRAINING_CONSTANTS.OUT_CHANNELS_OPTIONS))
        kernel_sizes.append(trial.suggest_categorical(f"kernel_size_l{i}",
                                                      TRAINING_CONSTANTS.KERNEL_SIZE_OPTIONS))

    # fully connected layers
    num_fc_layers = trial.suggest_int("num_fc_layers", TRAINING_CONSTANTS.MIN_FC_LAYERS,
                                      TRAINING_CONSTANTS.MAX_FC_LAYERS)
    fc_neurons_list = []

    for i in range(num_fc_layers):
        fc_neurons_list.append(trial.suggest_categorical(f"fc_neurons_l{i}", TRAINING_CONSTANTS.FC_NEURON_OPTIONS))


    # global hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", TRAINING_CONSTANTS.DROPOUT_RATE_MIN,
                                       TRAINING_CONSTANTS.DROPOUT_RATE_MAX)
    lr = trial.suggest_float("lr", TRAINING_CONSTANTS.LEARNING_RATE_MIN, TRAINING_CONSTANTS.LEARNING_RATE_MAX, log=True)

    model = DynamicConvNN(
        num_conv_layers=num_conv_layers,
        out_channels_list=out_channels_list,
        kernel_sizes=kernel_sizes,
        fc_neurons_list=fc_neurons_list,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=TRAINING_CONSTANTS.EARLY_STOPPING_PATIENCE,
                                  min_delta=TRAINING_CONSTANTS.MIN_DELTA)


    for epoch in range(1, TRAINING_CONSTANTS.MAXIMUM_EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # model evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # optuna pruning  if accuracy doesnt improtve
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # 9. Early Stopping
        early_stopper(accuracy, model)
        if early_stopper.early_stop:
            break

    return early_stopper.best_score if early_stopper.best_score is not None else accuracy


if __name__ == "__main__":
    """
    Entry point for hyperparameter tuning and final model training.

    Runs Optuna study to optimize hyperparameters for the convolutional neural network,
    prints the best parameters and accuracy, then trains the final model with the best parameters
    and saves it to disk.

    Parameters:
        None

    Returns:
        None
    """

    print("hyperparameter tuning by optuna")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRAINING_CONSTANTS.OPTUNA_TRIALS)

    print("\n" + "=" * 40)
    print("search finished")
    print("=" * 40)
    print(f"best precision: {study.best_value:.2f}%\n")

    best_params = study.best_params
    print("best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # training the final model with the best parameters
    print("\n" + "=" * 40)
    print("training the final model with the best parameters")
    print("=" * 40)

    num_conv_layers = best_params["num_conv_layers"]
    out_channels_list = [best_params[f"out_channels_l{i}"] for i in range(num_conv_layers)]
    kernel_sizes = [best_params[f"kernel_size_l{i}"] for i in range(num_conv_layers)]

    num_fc_layers = best_params["num_fc_layers"]
    fc_neurons_list = [best_params[f"fc_neurons_l{i}"] for i in range(num_fc_layers)]

    dropout_rate = best_params["dropout_rate"]
    lr = best_params["lr"]

    device, train_loader, val_loader, train_dataset = initialize_training()

    final_model = DynamicConvNN(
        num_conv_layers=num_conv_layers,
        out_channels_list=out_channels_list,
        kernel_sizes=kernel_sizes,
        fc_neurons_list=fc_neurons_list,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=lr)

    early_stopper = EarlyStopping(patience=TRAINING_CONSTANTS.EARLY_STOPPING_PATIENCE,
                                  min_delta=TRAINING_CONSTANTS.MIN_DELTA)

    stopped = False

    for epoch in range(1, TRAINING_CONSTANTS.MAXIMUM_EPOCHS + 1):
        final_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        final_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = final_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Final model - Epoch {epoch}/{TRAINING_CONSTANTS.MAXIMUM_EPOCHS}: Accuracy {accuracy:.2f}%")

        early_stopper(accuracy, final_model)
        if early_stopper.early_stop:
            stopped = True
            print(f"Early stopping triggered: {early_stopper.best_score:.2f}%")
            final_model.load_state_dict(early_stopper.best_weights)
            break

    if not stopped:
        print(f"Training stopped: {early_stopper.best_score:.2f}%")

        final_model.load_state_dict(early_stopper.best_weights)

    # save_path = "../data/models/dynamic_cnn_best_new_model.pth"
    save_path = TRAINING_CONSTANTS.MODEL_SAVE_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'hyperparameters': best_params,
        'state_dict': final_model.state_dict()
    }

    torch.save(checkpoint, save_path)

    print(f"\nBest model saved in: {save_path}")


