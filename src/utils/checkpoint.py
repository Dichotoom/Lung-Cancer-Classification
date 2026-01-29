import torch
import shutil
from pathlib import Path
from typing import Dict, Any

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path, filename: str = 'checkpoint.pth.tar'):
    """
    Saves a checkpoint and separately saves the best model.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = checkpoint_dir / filename
    torch.save(state, save_path)
    
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        shutil.copyfile(save_path, best_path)

def load_checkpoint(filepath: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Loads a checkpoint.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return checkpoint
