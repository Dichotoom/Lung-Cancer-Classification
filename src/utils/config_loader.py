import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Handles loading and accessing YAML configuration parameters."""
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to relative path from this file
            # src/utils/config_loader.py -> .../lung_cancer_classification/config/config.yaml
            root = Path(__file__).parent.parent.parent
            config_path = str(root / "config" / "config.yaml")

        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        # Create output directories
        root_dir = Path(__file__).parent.parent.parent
        for path_key in ['checkpoints', 'logs', 'visualizations']:
            rel_path = self.get(f'paths.{path_key}')
            if rel_path:
                full_path = root_dir / rel_path
                full_path.mkdir(parents=True, exist_ok=True)
    
    def __getattr__(self, name: str) -> Any:
        return self._config.get(name, None)
    
    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
            if value is None:
                return default
        return value

    def update(self, key_path: str, value: Any):
        """Updates a config value using dot notation key path."""
        keys = key_path.split('.')
        target = self._config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
