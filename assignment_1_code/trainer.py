import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        super().__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.wandb_logger = WandBLogger(
            enabled=True,
            model=model,
            run_name=f"{self.model.__class__.__name__}_{self.optimizer.__class__.__name__}_{self.loss_fn.__class__.__name__}",
        )

        #data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        self.best_val_pcacc = 0.0

        self.training_save_dir.mkdir(parents=True, exist_ok=True)

        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        self.model.train()
        self.train_metric.reset()

        total_loss = 0.0
        num_batches = 0

        for inputs, targets in tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.train_metric.update(outputs.detach(), targets)

            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = self.train_metric.accuracy()
        per_class_accuracy = self.train_metric.per_class_accuracy()

        print(f"\n______epoch {epoch_idx}")
        print(f"\n{self.train_metric}")

        for class_name in self.train_metric.classes:
            total_pred = self.train_metric.total_pred[class_name]
            correct_pred = self.train_metric.correct_pred[class_name]
            class_accuracy = (correct_pred / total_pred) if total_pred > 0 else 0.0
            print(f"Accuracy for class: {class_name:<6} is {class_accuracy:.2f}")

        return avg_loss, accuracy, per_class_accuracy

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        #eval mode
        self.model.eval()

        #reset metric
        self.val_metric.reset()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad(): #no gradients since we are validatinge
            for inputs, targets in tqdm(self.val_loader, desc=f"Validation Epoch {epoch_idx}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.val_metric.update(outputs.detach(), targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = self.val_metric.accuracy()
        per_class_accuracy = self.val_metric.per_class_accuracy()
        print(f"\n______epoch {epoch_idx}")
        print(f"\n{self.val_metric}")
        for class_name in self.val_metric.classes:
            total_pred = self.val_metric.total_pred[class_name]
            correct_pred = self.val_metric.correct_pred[class_name]
            class_accuracy = (correct_pred / total_pred) if total_pred > 0 else 0.0
            print(f"Accuracy for class: {class_name:<6} is {class_accuracy:.2f}")
        return avg_loss, accuracy, per_class_accuracy

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        self.best_val_pcacc = 0.0
        for epoch_idx in range(self.num_epochs):
            #train
            train_loss, train_acc, train_pcacc = self._train_epoch(epoch_idx)

            #log metrics
            self.wandb_logger.log(
                {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_per_class_accuracy": train_pcacc,
                }
            )

            #validate
            if (epoch_idx + 1) % self.val_frequency == 0 or epoch_idx == self.num_epochs - 1:
                val_loss, val_acc, val_pcacc = self._val_epoch(epoch_idx)
                self.wandb_logger.log(
                    {
                        "epoch": epoch_idx,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_per_class_accuracy": val_pcacc,
                    }
                )
                #save model if validation accuracy is better than previous best
                if val_pcacc > self.best_val_pcacc:
                    self.best_val_pcacc = val_pcacc
                    torch.save(self.model.state_dict(), self.training_save_dir / "best_model.pth")
                    print(f"Model saved at epoch {epoch_idx} with mPCAcc: {val_pcacc:.4f}")

            #update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


        self.wandb_logger.finish()     
