import argparse
import json
import os

from src.vit.transforms.transforms import get_transforms
from src.vit.data.data_loader import load_galaxy10
from src.vit.models.models import get_model
from src.vit.train.trainer import create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Vision Transformer")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta al archivo JSON de configuración (ej: configs/config.json)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Validar config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"No existe el archivo de config: {args.config}")

    # 2) Crear config centralizada
    config = json.load(open(args.config, 'r'))

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
    main()
