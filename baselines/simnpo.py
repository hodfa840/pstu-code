"""SimNPO: Simplified NPO without a reference model."""

import torch.nn.functional as F
from baselines.grad_diff import GradDiff
from baselines.trainer_utils import compute_batch_nll


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        loss_mask = forget_inputs["labels"] != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(
            self.beta * forget_loss).mean() * 2 / self.beta

        retain_inputs = {k: inputs["retain"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
