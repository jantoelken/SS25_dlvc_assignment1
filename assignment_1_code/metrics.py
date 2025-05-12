from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("prediction and Target must be a torch.Tensor")
        if len(prediction.shape) != 2:
            raise ValueError("Prediction must have shape (batchsize, n_classes)")
        if len(target.shape) != 1:
            raise ValueError("Target must have shape (batchsize,)")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Prediction and target must have the same batch size")
        #prediction must be aclass score vector
        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Prediction must be a class-score vector")
        if not torch.all(target >= 0) or not torch.all(target < len(self.classes)):
            raise ValueError("Target must have values between 0 and c-1 (true class labels).")
        
        #get the predicted class
        _, predicted = torch.max(prediction, 1)

        #update totals
        self.n_total += target.size(0)
        self.n_matching += (predicted == target).sum().item()

        #update per class accuracy
        for pred, target in zip(predicted, target):
            self.correct_pred[self.classes[pred.item()]] += (pred == target).item()
            self.total_pred[self.classes[target.item()]] += 1

    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        perc = self.accuracy() * 100
        per_class_perc = self.per_class_accuracy() * 100
        return f"Accuracy: {perc:.4f}%\nPer class accuracy: {per_class_perc:.4f}%"

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if self.n_total == 0:
            return 0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        class_accs = []
        for classname in self.classes:
            correct = self.correct_pred[classname]
            total = self.total_pred[classname]
            if total == 0:
                class_accs.append(0)
            else:
                class_accs.append(correct / total)
        return sum(class_accs) / len(class_accs)
