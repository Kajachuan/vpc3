from transformers import AutoModelForImageClassification

def get_model(checkpoint: str):
    """
    Permite cargar cualquier modelo de Huggingface para clasificación de imágenes.
    Ejemplos:
      - facebook/deit-small-patch16-224
      - facebook/deit-tiny-patch16-224
      - google/vit-base-patch16-224
      - microsoft/resnet-50
    """
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=10,
        ignore_mismatched_sizes=True,
    )
    return model
