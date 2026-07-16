import os
import subprocess
import yaml
from utils.logger import logger

class ConfigManager:
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._ensure_models_configured()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            logger.error(f"Config file {self.config_path} not found.")
            raise FileNotFoundError(f"Config file {self.config_path} not found.")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def _ensure_models_configured(self):
        """Automatically detect HEF models on the Pi if not set."""
        models_config = self.config.get('models', {})
        needs_save = False
        
        # Check if any model path is missing
        if not all(models_config.values()):
            logger.info("Some model paths are missing. Initiating automatic HEF discovery...")
            discovered_models = self._discover_hef_models()
            
            # Map discovered models based on keywords
            for path in discovered_models:
                filename = os.path.basename(path).lower()
                if "yolo" in filename and not models_config.get('yolo'):
                    self.config['models']['yolo'] = path
                    needs_save = True
                    logger.info(f"Auto-assigned YOLO model: {path}")
                elif "scrfd" in filename and not models_config.get('scrfd'):
                    self.config['models']['scrfd'] = path
                    needs_save = True
                    logger.info(f"Auto-assigned SCRFD model: {path}")
                elif "arcface" in filename and not models_config.get('arcface'):
                    self.config['models']['arcface'] = path
                    needs_save = True
                    logger.info(f"Auto-assigned ArcFace model: {path}")
                elif "repvgg" in filename or "reid" in filename and not models_config.get('reid'):
                    self.config['models']['reid'] = path
                    needs_save = True
                    logger.info(f"Auto-assigned Person ReID model: {path}")
            
            if needs_save:
                self._save_config()
                
        # Validate that all required models are now set
        missing = [k for k, v in self.config['models'].items() if not v]
        if missing:
            logger.error(f"Could not find HEF models for: {missing}. Please install them or update config.yaml manually.")
            raise FileNotFoundError(f"Missing required HEF models: {missing}")

    def _discover_hef_models(self):
        """Runs the find command to locate all .hef files on the system."""
        try:
            # We use a subprocess call to find all .hef files, discarding permission errors
            result = subprocess.run(['find', '/', '-name', '*.hef'], 
                                    stderr=subprocess.DEVNULL, 
                                    stdout=subprocess.PIPE, 
                                    text=True)
            paths = [p.strip() for p in result.stdout.split('\n') if p.strip()]
            logger.info(f"Discovered {len(paths)} HEF models on the system.")
            return paths
        except Exception as e:
            logger.error(f"Failed to discover HEF models: {e}")
            return []

    def get(self, section, key=None):
        if key:
            return self.config.get(section, {}).get(key)
        return self.config.get(section, {})
