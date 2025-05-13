## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from torchvision.models import resnet18
from pathlib import Path
import os

from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def train(args):

    ### Implement this function so that it trains a specific model as described in the instruction.md file
    ## feel free to change the code snippets given here, they are just to give you an initial structure
    ## but do not have to be used if you want to do it differently
    ## For device handling you can take a look at pytorch documentation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    conf = {
        "regularization": True,
        "augmentation": True,
        "lr_scheduler" : "cosine",
        "optimizer" : "adamw"
        }


    #Data Augmentation   
    train_transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
    if conf["augmentation"]:
        train_transform_list.insert(1, v2.RandomCrop(32, padding=4))
        train_transform_list.insert(1, v2.RandomHorizontalFlip(p=0.5))

    train_transform_list.append(
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )   
    train_transform = v2.Compose(train_transform_list)

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cifar10_dir = Path("/teamspace/studios/this_studio/cifar-10-batches-py")
    train_data = CIFAR10Dataset(
        cifar10_dir, Subset.TRAINING, transform=train_transform
    )
    val_data = CIFAR10Dataset(
        cifar10_dir, Subset.VALIDATION, transform=val_transform
    )

    #load resnet18
    basemodel = resnet18(weights=None)
    basemodel.fc = torch.nn.Linear(basemodel.fc.in_features, 10)


    model = DeepClassifier(basemodel)
    model.to(device)

    if conf["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.002,
            weight_decay=0.1
        )
    elif conf["optimizer"] == "adamw":
        if conf["regularization"]:
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=0.002, #bit higher since we use biger batch size 
                amsgrad=True,
                weight_decay=0.1
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=0.002, #bit higher since we use biger batch size 
                amsgrad=True
            )
    elif conf["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=0.01, #sgn needs higher lr
            momentum=0.9
        )
    loss_fn = torch.nn.CrossEntropyLoss()

    if conf["lr_scheduler"] == "exp":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif conf["lr_scheduler"] == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)


    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=512,  # feel free to change
        val_frequency=val_frequency,
        wab_suffix=f"_reg-{conf['regularization']}_aug-{conf['augmentation']}_lrscheduling-{conf['lr_scheduler']}_opt-{conf['optimizer']}"
    )
    trainer.train()


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)
