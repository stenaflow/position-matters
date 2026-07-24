from pathlib import Path

MODEL_NAME = "vit_base_patch16_224.mae"
MODEL_SLUG = "vit_base_patch16_224_mae"

_here        = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parents[2]          # token_reduction_privacy/
MODELS_ROOT  = PROJECT_ROOT / "saved_models" / "other_models" / MODEL_SLUG

LINEAR_PROBE_PATH            = MODELS_ROOT / "linear_probe.pth"
LINEAR_PROBE_LAST_BLOCK_PATH = MODELS_ROOT / "linear_probe_last_block.pth"
