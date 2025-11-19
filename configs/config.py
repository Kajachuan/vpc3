import json
import os
from typing import Tuple, List

class Config:
    """Configuración centralizada del proyecto"""
    
    # Valores por defecto
    DEFAULTS = {
        # Dataset
        "DATASET_NAME": "matthieulel/galaxy10_decals",
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42,
        "NUM_CLASSES": 10,
        
        # Modelo
        "MODEL_NAME": "google/vit-base-patch16-224-in21k",
        "PRETRAINED": True,
        "IMG_HEIGHT": 224,
        "IMG_WIDTH": 224,
        
        # Entrenamiento
        "BATCH_SIZE": 32,
        "EPOCHS": 50,
        "LEARNING_RATE": 0.001,
        "WEIGHT_DECAY": 1e-4,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        
        # Early Stopping
        "EARLY_STOPPING_PATIENCE": 10,
        "EARLY_STOPPING_MIN_DELTA": 0.01,
        
        # AMP y Optimización
        "USE_AMP": True,
        "AMP_DTYPE": "bfloat16",
        "GRADIENT_CLIP_VALUE": 1.0,
        
        # Augmentaciones
        "MORPH_KERNEL_SIZE": (7, 7),
        "ROTATION_DEGREES": 180,
        "TRANSLATE": (0.1, 0.1),
        "CONTRAST": 0.2,
        
        # Checkpointing
        "CHECKPOINT_DIR": "/Workspace/checkpoints",
        "SAVE_EVERY_N_STEPS": 100,
        
        # MLflow
        "EXPERIMENT_NAME": "/Shared/galaxy10_vit_classification",
        
        # Device
        "DEVICE": "cuda",
    }
    
    def __init__(self, config_path: str = None, **kwargs):
        """
        Inicializar configuración.
        
        Args:
            config_path (str, optional): Ruta al archivo JSON de configuración.
            **kwargs: Parámetros adicionales que sobrescriben los valores del JSON.
        """
        # Comenzar con valores por defecto
        self._config = self.DEFAULTS.copy()
        
        # Cargar desde JSON si existe
        if config_path and os.path.exists(config_path):
            self._load_from_json(config_path)
        
        # Sobrescribir con kwargs
        self._config.update(kwargs)
        
        # Convertir tuples de listas en JSON
        self._config["MORPH_KERNEL_SIZE"] = tuple(self._config.get("MORPH_KERNEL_SIZE", (7, 7)))
        self._config["TRANSLATE"] = tuple(self._config.get("TRANSLATE", (0.1, 0.1)))
    
    def _load_from_json(self, config_path: str):
        """Cargar configuración desde un archivo JSON"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Flatten el JSON (si viene con secciones anidadas)
            flat_config = self._flatten_dict(data)
            self._config.update(flat_config)
            print(f"✓ Configuración cargada desde: {config_path}")
        except FileNotFoundError:
            print(f"⚠ Archivo de configuración no encontrado: {config_path}")
        except json.JSONDecodeError as e:
            print(f"❌ Error al parsear JSON: {e}")
    
    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        """
        Flatten un diccionario anidado.
        
        Ejemplo:
            {"dataset": {"DATASET_NAME": "..."}} → {"DATASET_NAME": "..."}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Config._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __getattr__(self, name: str):
        """Permitir acceso a atributos como config.BATCH_SIZE"""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name in self._config:
            return self._config[name]
        
        raise AttributeError(f"Config no tiene atributo: {name}")
    
    def __setattr__(self, name: str, value):
        """Permitir establecer atributos como config.BATCH_SIZE = 64"""
        if name == '_config':
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_config'):
                super().__setattr__('_config', {})
            self._config[name] = value
    
    def to_dict(self) -> dict:
        """Convertir a diccionario para MLflow (flatten y convertir a strings)"""
        flat_config = {}
        for k, v in self._config.items():
            if isinstance(v, (list, tuple)):
                flat_config[k] = str(v)
            else:
                flat_config[k] = v
        return flat_config
    
    def to_json(self, output_path: str = None):
        """Guardar configuración actual a JSON"""
        if output_path is None:
            output_path = "configs/config_output.json"
        
        with open(output_path, 'w') as f:
            json.dump(self._config, f, indent=2)
        print(f"✓ Configuración guardada en: {output_path}")
    
    def __repr__(self) -> str:
        """Representación legible de la configuración"""
        lines = ["=" * 50, "CONFIG", "=" * 50]
        for key, value in sorted(self._config.items()):
            lines.append(f"  {key}: {value}")
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def print_config(self):
        """Imprimir configuración de forma legible"""
        print(self)