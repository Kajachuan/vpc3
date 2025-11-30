import argparse
import json
import os

from src.vit.transforms import get_transforms
from src.vit.data import load_galaxy10
from src.vit.models import get_model
from src.vit.train import create_trainer

def main(config_path: str):
    # 1) Validar config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No existe el archivo de config: {config_path}")

    # 2) Crear config centralizada
    config = json.load(open(config_path, 'r'))

    # 3) Transforms
    train_tf, test_tf = get_transforms(config)

    # 4) Dataset
    train_ds, test_ds = load_galaxy10(train_tf, test_tf)

    # 5) Modelo HF
    model = get_model(config["checkpoint"])

    # 6) Trainer (MLflow + métricas)
    trainer = create_trainer(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        config=config
    )

    # 7) Entrenar
    trainer.train()

    # 8) Evaluar
    trainer.evaluate()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
