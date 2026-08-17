"""Gradient Difference (GD): GA on forget set + retain regularization."""

import copy
from transformers import Trainer
from baselines.trainer_utils import compute_kl_divergence


class GradDiff(Trainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        ref = copy.deepcopy(model).to(self.accelerator.device)
        ref.eval()
        ref = self.accelerator.prepare_model(ref, evaluation_mode=True)
        return ref

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "KL":
            kl_loss, _ = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs)
            return kl_loss
        return model(**retain_inputs).loss

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_inputs = {k: inputs["forget"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = {k: inputs["retain"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
