#!/usr/bin/env python3
import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from src.utils.config_loader import Config
from src.utils.paths import PathManager
from src.utils.logging_utils import setup_logger
from src.data.data_loader import get_data_loaders
from src.models.classifier import get_model
from src.models.loss_functions import FocalLoss
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Override with CLI args
    if args.epochs:
        config.update('training.num_epochs', args.epochs)
    if args.batch_size:
        config.update('training.batch_size', args.batch_size)
        
    paths = PathManager(config)
    logger = setup_logger('Train', paths.get_log_path('training.log'))
    
    logger.info(f"Loaded configuration from {args.config}")
    
    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Data
    logger.info("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_dir=paths.get_data_dir(),
        class_names=config.get('data.class_names'),
        img_size=config.get('data.img_size'),
        mean=config.get('normalization.mean'),
        std=config.get('normalization.std'),
        batch_size=config.get('training.batch_size'),
        val_split=config.get('data.val_split'),
        num_workers=config.get('data.num_workers'),
        pin_memory=config.get('data.pin_memory'),
        use_weighted_sampler=config.get('training.use_weighted_sampler'),
        seed=config.get('project.seed')
    )
    
    # Model
    logger.info(f"Initializing {config.get('model.backbone')} model...")
    model = get_model(
        backbone_name=config.get('model.backbone'),
        num_classes=config.get('model.num_classes'),
        pretrained=config.get('model.pretrained'),
        dropout=config.get('model.dropout')
    )
    
    # Loss and Optimizer
    if config.get('loss.type') == 'focal':
        criterion = FocalLoss(
            alpha=config.get('loss.focal_alpha'),
            gamma=config.get('loss.focal_gamma')
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=paths.root / config.get('paths.checkpoints')
    )
    
    # Run
    trainer.fit(epochs=config.get('training.num_epochs'))

if __name__ == "__main__":
    main()
