import numpy as np
import mlflow

from transformers import TrainingArguments, Trainer
from src.vit.metrics.metrics import compute_metrics, log_confusion_matrix

class MLflowTrainer(Trainer):
    """
    Extiende HuggingFace Trainer para loguear automáticamente:
    - métricas a MLflow
    - confusion matrix al final del entrenamiento
    """
    def log_metrics(self, split, metrics):
        super().log_metrics(split, metrics)
        for k, v in metrics.items():
            mlflow.log_metric(f"{split}_{k}", v)

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)

        preds = np.argmax(output.predictions, axis=-1)
        labels = output.label_ids

        # confusion matrix
        class_names = list(range(output.predictions.shape[1]))
        log_confusion_matrix(labels, preds, class_names, self.args.output_dir)

        return output

def create_trainer(model, train_ds, test_ds, config):
    mlflow.set_experiment(config["mlflow_experiment"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        fp16=True,
        report_to=["mlflow"],
    )

    trainer = MLflowTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    return trainer
