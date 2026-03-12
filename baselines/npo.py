"""Negative Preference Optimization (NPO)."""

from baselines.grad_diff import GradDiff
from baselines.trainer_utils import compute_dpo_loss


class NPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model, ref_model=self.ref_model,
            win_inputs=None, lose_inputs=inputs["forget"],
            beta=self.beta,
        )
        retain_inputs = {k: inputs["retain"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
