"""Gradient Ascent (GA): maximize loss on the forget set."""

from transformers import Trainer


class GradAscent(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_inputs = inputs["forget"]
        outputs = model(
            input_ids=forget_inputs["input_ids"],
            attention_mask=forget_inputs["attention_mask"],
            labels=forget_inputs["labels"],
        )
        loss = -outputs.loss
        return (loss, outputs) if return_outputs else loss
