"""Architecture detection, GPU keepalive, and model configuration."""

import threading
import torch
import functools
from pathlib import Path

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model paths: check pstu_code/models/ first, fall back to parent project's models/
_MODELS_DIR = PROJECT_ROOT / "models"
if not _MODELS_DIR.exists():
    _MODELS_DIR = PROJECT_ROOT.parent / "models"

MODEL_CONFIGS = {
    "pythia-1.4b": {
        "infected_path": _MODELS_DIR / "pythia-1.4b-infected" / "final",
        "clean_model": "EleutherAI/pythia-1.4b",
    },
    "pythia-2.8b": {
        "infected_path": _MODELS_DIR / "pythia-2.8b-infected" / "final",
        "clean_model": "EleutherAI/pythia-2.8b",
    },
    "pythia-6.9b-gentle": {
        "infected_path": _MODELS_DIR / "pythia-6.9b-infected-gentle" / "final",
        "clean_model": "EleutherAI/pythia-6.9b",
    },
    "llama-3.1-8b-6ep": {
        "infected_path": _MODELS_DIR / "llama-3.1-8b-infected-6ep" / "final",
        "clean_model": "meta-llama/Llama-3.1-8B",
        "gradient_checkpointing": True,
        "fsdp_cls": "LlamaDecoderLayer",
    },
}

LUME_CONFIGS = {
    "1b": {
        "clean_model": "allenai/OLMo-1B-0724-hf",
        "infected_model": "llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
    },
    "7b": {
        "clean_model": "allenai/OLMo-7B-0724-Instruct-hf",
        "infected_model": "llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning",
    },
}


def detect_num_layers(state_dict):
    """Detect number of transformer layers from a state dict."""
    layer_nums = set()
    for name in state_dict:
        for part in name.split("."):
            if part.isdigit():
                layer_nums.add(int(part))
                break
    return max(layer_nums) + 1 if layer_nums else 0


def param_group(name, n_layers, group_size):
    """Map parameter name to group: 'embed', 'head', or 'g{i}'."""
    nl = name.lower()
    if "embed" in nl:
        return "embed"
    if "lm_head" in nl or "embed_out" in nl:
        return "head"
    for part in name.split("."):
        if part.isdigit():
            return f"g{int(part) // group_size}"
    return "g0"


class GPUKeepAlive:
    """Periodically runs a small CUDA kernel to prevent SLURM from killing
    low-usage jobs during long CPU-bound phases (e.g. Optuna overhead)."""

    def __init__(self, interval=30, device=None):
        self.interval = interval
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                a = torch.randn(256, 256, device=self.device)
                _ = a @ a
                del a
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
