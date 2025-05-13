import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path

from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.vit import MyViT
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cifar10_dir = Path("/teamspace/studios/this_studio/cifar-10-batches-py")
    train_data = CIFAR10Dataset(cifar10_dir, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(cifar10_dir, Subset.VALIDATION, transform=val_transform)

    vit_model = MyViT(
        image_size=32,
        patch_size=4,
        model_dim=128,
        num_heads=4,
        num_layers=4,
        num_classes=10
    )

    model = DeepClassifier(vit_model)
    model.to(device)

    # Fixed optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    trainer = ImgClassificationTrainer(
        model, optimizer, loss_fn, lr_scheduler,
        train_metric, val_metric, train_data, val_data,
        device, args.num_epochs, model_save_dir,
        batch_size=128, val_frequency=5, wab_suffix="_MyViT"
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Training ViT")
    args.add_argument("-d", "--gpu_id", default="0", type=str, help="index of which GPU to use")
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30
    train(args)