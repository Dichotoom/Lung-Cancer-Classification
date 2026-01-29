import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """
    Computes classification metrics.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        "classification_report": report,
        "confusion_matrix": cm
    }

@torch.no_grad()
def get_predictions(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions and probabilities from a loader.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

