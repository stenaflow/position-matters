import argparse
import json
import sys
from pathlib import Path

_here    = Path(__file__).resolve().parent
_mae_dir = _here.parent
sys.path.insert(0, str(_mae_dir))
sys.path.insert(0, str(_mae_dir.parents[2]))

from _config import MODEL_NAME, MODELS_ROOT, LINEAR_PROBE_LAST_BLOCK_PATH

import torch
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ClientVit, ServerVit
from models.position_predictor import ContextualPositionPredictor
from models.mae import TokenToImageDecoder
from models.decoder import AttackerDecoder
from utils.attack import SARA
from utils.evaluation import EvaluationTracker
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, SAMPLE_PCT

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",           type=int,   default=0)
parser.add_argument("--split",          type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--conf-threshold", type=float, default=0.0, dest="conf_threshold",
                    help="SARA confidence threshold (0 = disabled)")
_mode = parser.add_mutually_exclusive_group()
_mode.add_argument("--adv-only",        action="store_true", dest="adv_only")
_mode.add_argument("--pe-removal-only", action="store_true", dest="pe_removal_only")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"

if args.pe_removal_only:
    ENABLE_PE = False
    FILE_TAG  = "_pe_only"
elif args.adv_only:
    ENABLE_PE = True
    FILE_TAG  = "_adv_only"
else:
    ENABLE_PE = False
    FILE_TAG  = "_full"

_client_root = MODELS_ROOT / "adversarial" / "client"
_sara_root   = MODELS_ROOT / "adversarial" / "sara"


def _load_vit():
    vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE).eval()
    _lp = torch.load(LINEAR_PROBE_LAST_BLOCK_PATH, map_location=DEVICE)
    vit.head.load_state_dict(_lp["head_state_dict"])
    vit.blocks[-1].load_state_dict(_lp["last_block_state_dict"])
    vit.norm.load_state_dict(_lp["norm_state_dict"])
    return vit


def _load_client(vit, K):
    ckpt = torch.load(_client_root / f"s{K}{FILE_TAG}.pth", map_location=DEVICE)
    client = ClientVit(vit, K, enable_pos_embed=ENABLE_PE)
    client.load_state_dict(ckpt["client_state_dict"])
    return client.to(DEVICE).eval()


def _load_sara(K):
    pos_pred = ContextualPositionPredictor().to(DEVICE)
    _ckpt = torch.load(_sara_root / "tpp" / f"s{K}{FILE_TAG}.pth", map_location=DEVICE)
    pos_pred.load_state_dict(_ckpt.get("state_dict", _ckpt)); pos_pred.eval()
    mae = TokenToImageDecoder().to(DEVICE)
    mae.load_state_dict(
        torch.load(_sara_root / "mae" / f"s{K}{FILE_TAG}.pth", map_location=DEVICE)["decoder_state_dict"])
    mae.eval()
    decoder = AttackerDecoder().to(DEVICE)
    decoder.load_state_dict(
        torch.load(_sara_root / "decoder" / f"s{K}{FILE_TAG}.pth", map_location=DEVICE)["decoder_state_dict"])
    decoder.eval()
    return SARA(pos_pred, mae, decoder, conf_threshold=args.conf_threshold).to(DEVICE).eval()


def _shuffle_tokens(smashed):
    B = smashed.shape[0]
    cls_tok = smashed[:, :1, :]
    patches = smashed[:, 1:, :]
    N = patches.shape[1]
    perms = torch.stack([torch.randperm(N, device=smashed.device) for _ in range(B)])
    shuffled = patches[torch.arange(B, device=smashed.device).unsqueeze(1), perms]
    return torch.cat([cls_tok, shuffled], dim=1), perms


def _unshuffle_tokens(smashed_shuf, perms):
    B = smashed_shuf.shape[0]
    cls_tok = smashed_shuf[:, :1, :]
    patches = smashed_shuf[:, 1:, :]
    inv_perms = torch.argsort(perms, dim=1)
    unshuffled = patches[torch.arange(B, device=smashed_shuf.device).unsqueeze(1), inv_perms]
    return torch.cat([cls_tok, unshuffled], dim=1)


def run():
    dataset = get_imagenet_data(split="val", sample_pct=SAMPLE_PCT, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    ct_suffix   = f"_ct{args.conf_threshold}" if args.conf_threshold > 0 else ""
    results_dir = _here / "results" / f"shuffling{ct_suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for K in args.split:
        print(f"\n{'#'*60}\n  Adversarial — Shuffling  Split={K}  mode={FILE_TAG}  [{MODEL_NAME}]\n{'#'*60}")

        vit    = _load_vit()
        client = _load_client(vit, K)
        server = ServerVit(vit=vit, split_block_number=K).to(DEVICE).eval()
        sara   = _load_sara(K)
        tracker = EvaluationTracker()

        out_path = results_dir / f"s{K}{FILE_TAG}.json"
        if out_path.exists():
            print(f"  Already exists → {out_path}")
        else:
            tracker.reset()
            for x, y in tqdm(loader, desc=f"s{K}{FILE_TAG} shuffle"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    smashed = client(x)
                    smashed_shuf, perms = _shuffle_tokens(smashed)
                    smashed_orig = _unshuffle_tokens(smashed_shuf, perms)
                    pred = vit.head(server(smashed_orig)[:, 0]).argmax(dim=1)
                    rec  = sara(smashed_shuf)
                tracker.update(x, rec, pred, y)

            label  = f"Adversarial Shuffling [SARA] [{MODEL_NAME}] split={K} mode={FILE_TAG}"
            result = tracker.result(label, split_block=K)
            print(f"  Acc={result['accuracy']:.4f}  SSIM={result['ssim']:.4f}  "
                  f"PSNR={result['psnr']:.2f}  F-SIM={result['fsim']:.4f}")
            out_path.write_text(json.dumps(result, indent=4))
            print(f"\nResults saved → {out_path}")

        del vit, client, server, sara
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
