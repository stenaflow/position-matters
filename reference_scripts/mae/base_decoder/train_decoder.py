import sys
from pathlib import Path

_here    = Path(__file__).resolve().parent
_mae_dir = _here.parent
sys.path.insert(0, str(_mae_dir))
sys.path.insert(0, str(_mae_dir.parents[1]))

from _config import MODEL_NAME, MODELS_ROOT

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

from models.split_vit import ClientVit
from models.decoder import AttackerDecoder
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, TRAIN_SAMPLE_PCT, VAL_SAMPLE_PCT
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",       type=int,   default=0)
parser.add_argument("--split",      type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--epochs",     type=int,   default=20)
parser.add_argument("--patience",   type=int,   default=5)
parser.add_argument("--min-delta",  type=float, default=0.001, dest="min_delta")
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--sample-pct", type=float, default=TRAIN_SAMPLE_PCT, dest="sample_pct")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()


def main():
    train_ds = get_imagenet_data(split="train", sample_pct=args.sample_pct)
    val_ds   = get_imagenet_data(split="val",   sample_pct=VAL_SAMPLE_PCT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

    for K in args.split:
        print(f"\n{'='*60}\n  Training decoder  split={K}  [{MODEL_NAME}]\n{'='*60}")

        save_dir = MODELS_ROOT / "sara" / "decoder"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"s{K}.pth"

        vit = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE).eval()
        vit.requires_grad_(False)
        client = ClientVit(vit, split_block_number=K).to(DEVICE).eval()

        decoder   = AttackerDecoder().to(DEVICE)
        criterion = nn.MSELoss()
        opt       = torch.optim.AdamW(decoder.parameters(), lr=args.lr)
        scaler    = torch.amp.GradScaler("cuda", enabled=AMP)
        sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_val   = float("inf")
        no_improve = 0

        for ep in range(1, args.epochs + 1):
            decoder.train()
            t_loss = 0.0
            for x, _ in tqdm(train_dl, desc=f"  s{K} ep {ep}/{args.epochs} [train]"):
                x = x.to(DEVICE, non_blocking=True)
                with torch.no_grad():
                    smashed = client(x)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=AMP):
                    loss = criterion(decoder(smashed), x)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                t_loss += loss.item()
            sched.step()
            train_loss = t_loss / len(train_dl)

            decoder.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, _ in tqdm(val_dl, desc=f"  s{K} ep {ep}/{args.epochs} [val  ]"):
                    x = x.to(DEVICE, non_blocking=True)
                    smashed = client(x)
                    with torch.amp.autocast("cuda", enabled=AMP):
                        v_loss += criterion(decoder(smashed), x).item()
            val_loss = v_loss / len(val_dl)
            print(f"  Epoch {ep}: train={train_loss:.6f}  val={val_loss:.6f}")

            ckpt = {"decoder_state_dict": decoder.state_dict(),
                    "split_block": K, "epoch": ep, "val_loss": val_loss,
                    "args": vars(args)}

            if val_loss < best_val - args.min_delta:
                best_val = val_loss; no_improve = 0
                torch.save(ckpt, save_path)
                print(f"    → best (val={best_val:.6f})")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    early stopping at epoch {ep}"); break

        print(f"  Saved → {save_path}")
        del vit, client, decoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
