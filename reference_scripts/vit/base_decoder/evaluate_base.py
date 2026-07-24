import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
from piq import psnr, fsim

from models.split_vit import ClientVit, ServerVit
from models.decoder import AttackerDecoder
from utils.image import get_imagenet_data
from utils.reconstruction import compute_ssim
from config import SPLIT_BLOCK, BATCH_SIZE, SAMPLE_PCT, IMAGENET_MEAN, IMAGENET_STD

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",  type=int, default=0)
parser.add_argument("--split", type=int, nargs="+", default=[SPLIT_BLOCK], metavar="K")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

_script_dir   = Path(__file__).resolve().parent
_project_root = _script_dir.parents[1]

_MEAN = torch.tensor(IMAGENET_MEAN)
_STD  = torch.tensor(IMAGENET_STD)


def _denormalize(x):
    mean = _MEAN.to(x.device).view(1, 3, 1, 1)
    std  = _STD.to(x.device).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def run():
    results_dir = _script_dir / "results" / "base"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_imagenet_data(split="val", sample_pct=SAMPLE_PCT, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    for K in args.split:
        print(f"\n{'='*60}\n  split={K}\n{'='*60}")

        dec_path = _project_root / f"saved_models/decoder/s{K}.pth"
        if not dec_path.exists():
            print(f"  [skip] Decoder not found: {dec_path}"); continue

        vit    = timm.create_model("vit_base_patch16_224", pretrained=True).to(DEVICE).eval()
        client = ClientVit(vit=vit, split_block_number=K).to(DEVICE).eval()
        server = ServerVit(vit=vit, split_block_number=K).to(DEVICE).eval()
        decoder = AttackerDecoder().to(DEVICE)
        decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE)["decoder_state_dict"])
        decoder.eval()

        correct = total = 0
        ssim_sum = psnr_sum = fsim_sum = 0.0

        for x, y in tqdm(loader, desc=f"  s{K}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                smashed  = client(x)
                pred     = vit.head(server(smashed)[:, 0]).argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
                rec  = decoder(smashed)
                x_dn = _denormalize(x); r_dn = _denormalize(rec)
                for i in range(x.size(0)):
                    ssim_sum += float(compute_ssim(x_dn[i], r_dn[i]))
                    psnr_sum += float(psnr(r_dn[i:i+1], x_dn[i:i+1], data_range=1.0))
                    fsim_sum += float(fsim(r_dn[i:i+1], x_dn[i:i+1], data_range=1.0))

        result = {
            "split_block": K,
            "label":       f"Base decoder (split={K})",
            "accuracy":    correct / total,
            "ssim":        ssim_sum / total,
            "psnr":        psnr_sum / total,
            "fsim":        fsim_sum / total,
            "n_samples":   total,
            "timestamp":   datetime.now().isoformat(),
        }
        print(f"  Acc={result['accuracy']:.4f}  SSIM={result['ssim']:.4f}  "
              f"PSNR={result['psnr']:.2f}  FSIM={result['fsim']:.4f}")

        out_path = results_dir / f"s{K}.json"
        out_path.write_text(json.dumps(result, indent=4))
        print(f"  Saved → {out_path}")

        del vit, client, server, decoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
