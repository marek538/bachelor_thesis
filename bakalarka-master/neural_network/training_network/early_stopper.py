import copy


class EarlyStopping:
    """
    Implements early stopping to terminate training when a monitored metric stops improving.

    Attributes:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        counter (int): Counts epochs with no improvement.
        best_score (float or None): Best score observed so far.
        early_stop (bool): Flag indicating whether training should be stopped.
        best_weights (dict or None): Model weights corresponding to the best score.

    Methods:
        __init__(patience=5, min_delta=0):
            Initializes the early stopping object with specified patience and minimum delta.

        __call__(current_score, model):
            Checks if the current score is an improvement. Updates internal state and model weights.
            Sets early_stop to True if no improvement for 'patience' epochs.

            Parameters:
                current_score (float): The current value of the monitored metric (e.g., validation loss or accuracy).
                model (torch.nn.Module): The model being trained.

            Returns:
                None. Updates internal state and best_weights.
    """

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
