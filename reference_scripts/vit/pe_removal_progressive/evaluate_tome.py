import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ServerVit
from models.tome_vit import ClientVitToMe
from models.position_predictor import ContextualPositionPredictor
from models.mae import TokenToImageDecoder
from models.decoder import AttackerDecoder
from utils.attack import SARA
from utils.evaluation import EvaluationTracker
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, SAMPLE_PCT

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",        type=int,   default=0)
parser.add_argument("--split",       type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--r-values",    type=int,   nargs="+", default=list(range(5, 95, 5)),
                    dest="r_values", metavar="R")
parser.add_argument("--loss",           type=str,   default="kl", choices=["ce", "kl"])
parser.add_argument("--temperature",    type=float, default=4.0)
parser.add_argument("--conf-threshold", type=float, default=0.0, dest="conf_threshold",
                    help="SARA confidence threshold (0 = disabled)")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

_script_dir   = Path(__file__).resolve().parent
_project_root = _script_dir.parents[1]
_client_root  = _project_root / "saved_models" / "pe_removal_progressive" / "client"
_sara_root    = _project_root / "saved_models" / "pe_removal_progressive" / "sara"


def _tag(K):
    return f"s{K}_{args.loss}" + (f"_T{args.temperature}" if args.loss == "kl" else "")


def _load_client_tome(K):
    ckpt = torch.load(_client_root / f"{_tag(K)}.pth", map_location=DEVICE)
    vit  = timm.create_model("vit_base_patch16_224", pretrained=True)
    client = ClientVitToMe(vit, K, enable_pos_embed=False)
    client.load_state_dict(ckpt["client_state_dict"], strict=False)
    return client.to(DEVICE).eval()


def _load_sara(K):
    tag = _tag(K)
    pos_pred = ContextualPositionPredictor().to(DEVICE)
    _ckpt = torch.load(_sara_root / "tpp" / f"{tag}.pth", map_location=DEVICE)
    pos_pred.load_state_dict(_ckpt.get("state_dict", _ckpt)); pos_pred.eval()
    mae = TokenToImageDecoder().to(DEVICE)
    mae.load_state_dict(torch.load(_sara_root / "mae" / f"{tag}.pth", map_location=DEVICE)["decoder_state_dict"]); mae.eval()
    decoder = AttackerDecoder().to(DEVICE)
    decoder.load_state_dict(torch.load(_sara_root / "decoder" / f"{tag}.pth", map_location=DEVICE)["decoder_state_dict"]); decoder.eval()
    return SARA(pos_pred, mae, decoder, conf_threshold=args.conf_threshold).to(DEVICE).eval()


def run():
    dataset = get_imagenet_data(split="val", sample_pct=SAMPLE_PCT, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    ct_suffix   = f"_ct{args.conf_threshold}" if args.conf_threshold > 0 else ""
    results_dir = _script_dir / "results" / f"tome{ct_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for K in args.split:
        print(f"\n{'#'*60}\n  PE-Removal Progressive — Split={K}\n{'#'*60}")

        # Separate vit instances: client gets ToMe-patched, server must stay untouched.
        vit_c  = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        vit_s  = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        client = _load_client_tome(K)
        server = ServerVit(vit=vit_s, split_block_number=K).to(DEVICE).eval()
        sara   = _load_sara(K)
        tracker = EvaluationTracker()

        out_path = results_dir / f"{_tag(K)}.json"
        existing = {}
        if out_path.exists():
            for rec in json.loads(out_path.read_text()):
                existing[rec["r"]] = rec

        for r in args.r_values:
            approx_tokens = max(197 - K * r, 1)
            label = f"PE-Removal Progressive ToMe r={r} (~{approx_tokens} tokens) [SARA] split={K}"
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
                  f"PSNR={result['psnr']:.2f}  F-SIM={result['fsim']:.4f}")
            existing[r] = result
            json.dump(sorted(existing.values(), key=lambda x: x["r"]), open(out_path, "w"), indent=4)

        print(f"\nResults saved → {out_path}")

        del vit_c, vit_s, client, server, sara
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
