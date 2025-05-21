from transformers import (
    EvalPrediction,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import numpy as np
import torchmetrics
import torch


class StreamingMetricsCallback(TrainerCallback):
    def __init__(self, ignore_index: int = -100):

        self.ignore_index = ignore_index
        self.acc = None

    def on_train_begin(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        model = kwargs["model"]
        self.acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=model.config.vocab_size,
            average="micro",
            ignore_index=self.ignore_index,
        ).to(model.device)

    def on_prediction_step(
        self, args, state: TrainerState, control: TrainerControl, **kwargs
    ):
        if control.should_evaluate:
            logits = kwargs["outputs"].logits
            labels = kwargs["inputs"]["labels"]
            preds = logits.argmax(dim=-1)

            mask = labels.ne(self.ignore_index)
            self.acc(preds[mask], labels[mask])
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        kwargs["metrics"]["eval_masked_accuracy"] = self.acc.compute().item()
        self.acc.reset()
        return control


def masking_accuracy(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = (preds[mask] == labels[mask]).mean()
    return {
        "masked_accuracy": acc,
    }
