from pathlib import Path

MODEL_NAME = "vit_base_patch16_224.mae"
MODEL_SLUG = "vit_base_patch16_224_mae"

_here        = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parents[1]          # position-matters/
MODELS_ROOT  = PROJECT_ROOT / "saved_models" / "mae"

LINEAR_PROBE_PATH            = MODELS_ROOT / "linear_probe.pth"
LINEAR_PROBE_LAST_BLOCK_PATH = MODELS_ROOT / "linear_probe_last_block.pth"
