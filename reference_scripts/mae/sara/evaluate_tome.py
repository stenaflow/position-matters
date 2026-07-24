import sys
from pathlib import Path

_here    = Path(__file__).resolve().parent
_mae_dir = _here.parent
sys.path.insert(0, str(_mae_dir))
sys.path.insert(0, str(_mae_dir.parents[2]))

from _config import MODEL_NAME, MODELS_ROOT, PROJECT_ROOT, LINEAR_PROBE_LAST_BLOCK_PATH

import json
import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from models.tome_vit import ClientVitToMe
from models.split_vit import ServerVit
from utils.attack import load_sara
from utils.evaluation import EvaluationTracker
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, SAMPLE_PCT

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",   type=int,   default=0)
parser.add_argument("--split",  type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--r-values", type=int, nargs="+",
                    default=list(range(5, 95, 5)), dest="r_values", metavar="R",
                    help="r values to evaluate (default: 5 10 … 90)")
parser.add_argument("--conf-threshold", type=float, default=0.0, dest="conf_threshold",
                    help="SARA confidence threshold (0 = disabled)")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"


def run():
    dataset = get_imagenet_data(split="val", sample_pct=SAMPLE_PCT, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    ct_suffix   = f"_ct{args.conf_threshold}" if args.conf_threshold > 0 else ""
    results_dir = _here / "results" / f"tome{ct_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for K in args.split:
        print(f"\n{'#'*60}\n  ToMe + SARA  split={K}  [{MODEL_NAME}]\n{'#'*60}")

        vit_c  = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE).eval()
        vit_s  = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE).eval()
        _lp = torch.load(LINEAR_PROBE_LAST_BLOCK_PATH, map_location=DEVICE)
        vit_s.head.load_state_dict(_lp["head_state_dict"])
        vit_s.blocks[-1].load_state_dict(_lp["last_block_state_dict"])
        vit_s.norm.load_state_dict(_lp["norm_state_dict"])
        client = ClientVitToMe(vit=vit_c, split_block_number=K).to(DEVICE).eval()
        server = ServerVit(vit=vit_s, split_block_number=K).to(DEVICE).eval()

        sara    = load_sara(K, DEVICE, PROJECT_ROOT, models_root=MODELS_ROOT,
                            conf_threshold=args.conf_threshold)
        tracker = EvaluationTracker()

        out_path = results_dir / f"s{K}.json"
        existing = {}
        if out_path.exists():
            for rec in json.loads(out_path.read_text()):
                existing[rec["r"]] = rec

        for r in args.r_values:
            approx_tokens = max(197 - K * r, 1)
            if approx_tokens < 1:
                break
            label = f"ToMe r={r} (~{approx_tokens} tokens) [SARA] [{MODEL_NAME}] split={K}"
            print(f"\n{'='*60}\n  {label}\n{'='*60}")

            tracker.reset()
            for x, y in tqdm(loader, desc=f"s{K} r={r}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    smashed, _, _ = client(x, r)
                    pred = vit_s.head(server(smashed)[:, 0]).argmax(dim=1)
                    rec  = sara(smashed)
                tracker.update(x, rec, pred, y)

            result = tracker.result(label, r=r, approx_tokens=approx_tokens, split_block=K)
            print(f"  Acc={result['accuracy']:.4f}  SSIM={result['ssim']:.4f}  "
                  f"PSNR={result['psnr']:.2f}  FSIM={result['fsim']:.4f}")
            existing[r] = result
            merged = sorted(existing.values(), key=lambda x: x["r"])
            json.dump(merged, open(out_path, "w"), indent=4)

        print(f"\nResults saved → {out_path}")
        del vit_c, vit_s, client, server, sara
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
