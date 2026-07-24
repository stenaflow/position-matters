import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ClientVit, ServerVit
from utils.attack import load_sara
from utils.evaluation import EvaluationTracker
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, SAMPLE_PCT

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",           type=int,   default=0)
parser.add_argument("--split",          type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--conf-threshold", type=float, default=0.0, dest="conf_threshold",
                    help="SARA confidence threshold (0 = disabled)")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

_script_dir   = Path(__file__).resolve().parent
_project_root = _script_dir.parents[1]


def _shuffle_tokens(smashed: torch.Tensor):
    B, T, D = smashed.shape
    cls_tok = smashed[:, :1, :]
    patches = smashed[:, 1:, :]
    N = patches.shape[1]
    perms = torch.stack([torch.randperm(N, device=smashed.device) for _ in range(B)])
    shuffled = patches[torch.arange(B, device=smashed.device).unsqueeze(1), perms]
    return torch.cat([cls_tok, shuffled], dim=1), perms


def _unshuffle_tokens(smashed_shuf: torch.Tensor, perms: torch.Tensor):
    B, T, D = smashed_shuf.shape
    cls_tok = smashed_shuf[:, :1, :]
    patches = smashed_shuf[:, 1:, :]
    inv_perms = torch.argsort(perms, dim=1)
    unshuffled = patches[torch.arange(B, device=smashed_shuf.device).unsqueeze(1), inv_perms]
    return torch.cat([cls_tok, unshuffled], dim=1)


def run():
    dataset = get_imagenet_data(split="val", sample_pct=SAMPLE_PCT, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    ct_suffix   = f"_ct{args.conf_threshold}" if args.conf_threshold > 0 else ""
    results_dir = _script_dir / "results" / f"shuffling{ct_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for K in args.split:
        print(f"\n{'#'*60}")
        print(f"  Split block = {K}  (Token Shuffle + SARA)")
        print(f"{'#'*60}")

        vit    = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        client = ClientVit(vit=vit, split_block_number=K).to(DEVICE).eval()
        server = ServerVit(vit=vit, split_block_number=K).to(DEVICE).eval()

        sara    = load_sara(K, DEVICE, _project_root, conf_threshold=args.conf_threshold)
        tracker = EvaluationTracker()

        for x, y in tqdm(loader, desc=f"s{K}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                smashed = client(x)
                smashed_shuf, perms = _shuffle_tokens(smashed)
                # server un-shuffles with the shared key → normal accuracy
                smashed_orig = _unshuffle_tokens(smashed_shuf, perms)
                pred = vit.head(server(smashed_orig)[:, 0]).argmax(dim=1)
                # attacker only sees shuffled tokens
                rec = sara(smashed_shuf)
            tracker.update(x, rec, pred, y)

        result = tracker.result(label=f"Shuffle SARA split={K}", split_block=K)
        print(f"  Acc={result['accuracy']:.4f}  SSIM={result['ssim']:.4f}  "
              f"PSNR={result['psnr']:.2f}  FSIM={result['fsim']:.4f}")

        out_path = results_dir / f"s{K}.json"
        out_path.write_text(json.dumps(result, indent=4))
        print(f"Results saved → {out_path}")

        del vit, client, server, sara
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
