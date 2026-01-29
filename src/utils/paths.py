from pathlib import Path
from src.utils.config_loader import Config

class PathManager:
    """Centralizes path management for checkpoints, logs, and data."""
    def __init__(self, config: Config):
        self.config = config
        # Root is assumed to be where setup.py is (parent of src)
        self.root = Path(__file__).parent.parent.parent
        
    def get_checkpoint_path(self, filename: str) -> Path:
        return self.root / self.config.get('paths.checkpoints') / filename
    
    def get_visualization_path(self, filename: str) -> Path:
        return self.root / self.config.get('paths.visualizations') / filename
    
    def get_log_path(self, filename: str) -> Path:
        return self.root / self.config.get('paths.logs') / filename

    def get_data_dir(self) -> Path:
        return self.root / self.config.get('data.root_dir')
