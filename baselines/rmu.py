"""Representation Misdirection Unlearning (RMU).

Steers internal activations on the forget set towards a random control vector,
while preserving retain-set activations via MSE loss.
"""

import re
import torch
import torch.nn.functional as F
from baselines.grad_diff import GradDiff


class RMU(GradDiff):
    def __init__(self, module_regex="model\\.layers\\.7",
                 trainable_params_regex=None, steering_coeff=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if trainable_params_regex is None:
            trainable_params_regex = [
                "model\\.layers\\.(5|6|7)\\.mlp\\.down_proj\\.weight"]
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        self.trainable_params_regex = trainable_params_regex
        self.module_regex = module_regex
        self.model_module = self._get_matching_module(self.model, module_regex)
        self.ref_module = self._get_matching_module(self.ref_model, module_regex)
        self.steering_coeff = steering_coeff
        self.control_vec = None

    def create_optimizer(self):
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        super().create_optimizer()
        self._freeze_all_params(self.model, True)

    def _get_matching_module(self, model, module_regex):
        matched = {name: mod for name, mod in model.named_modules()
                   if re.fullmatch(module_regex, name)}
        if len(matched) != 1:
            raise ValueError(
                f"Expected exactly 1 module matching '{module_regex}', "
                f"got {len(matched)}: {list(matched.keys())}")
        return next(iter(matched.values()))

    def _freeze_all_params(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def _set_trainable_params(self, model, patterns, requires_grad=True):
        for name, p in model.named_parameters():
            if any(re.fullmatch(pat, name) for pat in patterns):
                p.requires_grad = requires_grad

    def _forward_with_cache(self, model, inputs, module, no_grad=True):
        cache = []
        def hook(mod, inp, out):
            cache.append(out[0] if isinstance(out, tuple) else out)
        handle = module.register_forward_hook(hook)
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        handle.remove()
        return cache[0], outputs

    def _get_control_vector(self, dim):
        if self.control_vec is None:
            rv = torch.rand(1, 1, dim)
            self.control_vec = rv / torch.norm(rv) * self.steering_coeff
        return self.control_vec

    def _activation_loss(self, act1, act2, mask):
        diff = F.mse_loss(act1, act2, reduction="none")
        mask_exp = mask.unsqueeze(-1).expand_as(diff)
        per_seq = (diff * mask_exp).mean(dim=2).sum(dim=1)
        n_tokens = mask.sum(dim=-1, keepdim=True)
        return (per_seq / n_tokens).mean()

    def compute_retain_loss(self, model, retain_inputs):
        if self.retain_loss_type == "EMBED_DIFF":
            model_act, _ = self._forward_with_cache(
                model, retain_inputs, self.model_module, no_grad=False)
            ref_act, _ = self._forward_with_cache(
                self.ref_model, retain_inputs, self.ref_module, no_grad=True)
            mask = retain_inputs["labels"] != -100
            return self._activation_loss(model_act, ref_act.to(model_act.device), mask)
        return super().compute_retain_loss(model, retain_inputs)

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_inputs = {k: inputs["forget"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        model_act, forget_outputs = self._forward_with_cache(
            model, forget_inputs, self.model_module, no_grad=False)

        cv = self._get_control_vector(model_act.shape[-1])
        cv = cv.to(dtype=model_act.dtype, device=model_act.device)
        cv = cv.expand_as(model_act)
        mask = forget_inputs["labels"] != -100
        forget_loss = self._activation_loss(model_act, cv, mask)

        retain_inputs = {k: inputs["retain"][k]
                         for k in ("input_ids", "attention_mask", "labels")}
        retain_loss = self.compute_retain_loss(model, retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
