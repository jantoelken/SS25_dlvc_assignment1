## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os
from pathlib import Path
from tqdm import tqdm

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    cifar10_dir = Path("/teamspace/studios/this_studio/cifar-10-batches-py")
    test_data = CIFAR10Dataset(
        cifar10_dir, Subset.TEST, transform=transform
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=512, shuffle=False
    )

    num_test_data = len(test_data)

    basemodel = resnet18(weights=None)
    basemodel.fc = torch.nn.Linear(basemodel.fc.in_features, 10)
    model = DeepClassifier(basemodel)
    model.load(args.path_to_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss
    ### and metrics to terminal after testing is finished
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # No gradient computation needed for testing
        for inputs, targets in tqdm(test_data_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Update metrics
            test_metric.update(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = test_metric.accuracy()
    per_class_accuracy = test_metric.per_class_accuracy()

    # Print results
    print(f"\nModel: {Path(args.path_to_trained_model).name}")
    print(f"\ntest loss: {avg_loss}")
    print(f"\naccuracy: {accuracy:.4f}")
    print(f"per class accuracy: {per_class_accuracy:.4f}")
    
    for class_name in test_data.classes:
        total_pred = test_metric.total_pred[class_name]
        correct_pred = test_metric.correct_pred[class_name]
        class_accuracy = (correct_pred / total_pred) if total_pred > 0 else 0.0
        print(f"Accuracy for class: {class_name:<6} is {class_accuracy:.2f}")


if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Testing")
    args.add_argument(
        "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    )
    args.add_argument(
        "-p", "--path_to_trained_model", 
        default="/teamspace/studios/this_studio/SS25_dlvc_assignment1/saved_models/resnet18_reg-False_aug-False_lrscheduling-exp_opt-adamw.pth", 
        type=str, 
        help="path to the trained model weights"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    test(args)
