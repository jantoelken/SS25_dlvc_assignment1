import torch
import torch.nn as nn
from pathlib import Path

class DeepClassifier(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)
    

    def save(self, save_dir: Path, suffix=None):
        '''
        Saves the model, adds suffix to filename if given
        '''
        # Ensure directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if suffix is not None:
            filename = f"model_{suffix}.pth"
        else:
            filename = f"model.pth"

        save_path = save_dir / filename
        # Save the state_dict of self.net (the actual model)
        torch.save(self.net.state_dict(), save_path)

    def load(self, path):
        '''
        Loads model from path
        Does not work with transfer model
        '''
        
        if Path(path).is_file():
            self.net.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        else:
            raise FileNotFoundError(f"No model file found at{path}")