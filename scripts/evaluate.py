#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from src.utils.config_loader import Config
from src.utils.paths import PathManager
from src.utils.logging_utils import setup_logger
from src.utils.visualization import Visualizer
from src.utils.checkpoint import load_checkpoint
from src.data.data_loader import get_data_loaders
from src.models.classifier import get_model
from src.evaluation.metrics import get_predictions, compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    
    config = Config(args.config)
    paths = PathManager(config)
    logger = setup_logger('Evaluate', paths.get_log_path('evaluation.log'))
    visualizer = Visualizer(output_dir=paths.root / config.get('paths.visualizations'))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data (Validation only)
    _, val_loader = get_data_loaders(
        data_dir=paths.get_data_dir(),
        class_names=config.get('data.class_names'),
        img_size=config.get('data.img_size'),
        mean=config.get('normalization.mean'),
        std=config.get('normalization.std'),
        batch_size=config.get('training.batch_size'),
        val_split=config.get('data.val_split'),
        num_workers=config.get('data.num_workers'),
        pin_memory=config.get('data.pin_memory'),
        seed=config.get('project.seed')
    )
    
    # Model
    model = get_model(
        backbone_name=config.get('model.backbone'),
        num_classes=config.get('model.num_classes'),
        pretrained=config.get('model.pretrained'),
        dropout=config.get('model.dropout')
    ).to(device)
    
    # Load Best Model
    checkpoint_path = paths.get_checkpoint_path('best_model.pth')
    try:
        load_checkpoint(checkpoint_path, model)
        logger.info(f"Loaded best model from {checkpoint_path}")
    except FileNotFoundError:
        logger.error(f"Checkpoint not found at {checkpoint_path}. Train model first.")
        return

    # Predictions
    logger.info("Generating predictions...")
    y_true, y_pred, y_probs = get_predictions(model, val_loader, device)
    
    # Metrics
    metrics = compute_metrics(y_true, y_pred, config.get('data.class_names'))
    
    # Visualization
    cm_path = visualizer.plot_confusion_matrix(metrics['confusion_matrix'], config.get('data.class_names'))
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    roc_path = visualizer.plot_roc_curves(y_true, y_probs, config.get('data.class_names'))
    logger.info(f"ROC curves saved to: {roc_path}")

if __name__ == "__main__":
    main()
