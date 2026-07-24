# Reference Scripts

This folder contains only the training and evaluation scripts corresponding to the checkpoints used by `demo.ipynb`.

The `.pth` files themselves are not included in `position-matters/` because they are too large for the repository.
These scripts are included so the checkpoints can be regenerated.

Included groups:

- `vit/base_decoder/`: reference scripts for the baseline decoder checkpoints used in the `Baseline` column
- `vit/sara/`: SARA training and reduction-evaluation scripts used for the attack setting
- `vit/pe_removal_progressive/`: client/SARA training and reduction-evaluation scripts for the PE-Removal Progressive defense
- `vit/adversarial/`: client/SARA training and reduction-evaluation scripts for the adversarial defense
- `mae/...`: the matching MAE-B/16 variants used when `MODEL = "mae"`

Notes:

- These are copied from the main project for provenance only.
- They are not wired into the notebook execution path.
- To regenerate checkpoints, use the training scripts first and then place the resulting files into `position-matters/saved_models/` with the layout expected by the notebook.
- `mae/_config.py` is included because the MAE-side reference scripts depend on it.
