import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parents[1]))

from _config import MODEL_NAME, MODELS_ROOT, LINEAR_PROBE_LAST_BLOCK_PATH

import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from utils.image import get_imagenet_data
from config import BATCH_SIZE, TRAIN_SAMPLE_PCT, VAL_SAMPLE_PCT

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",       type=int,   default=0)
parser.add_argument("--epochs",     type=int,   default=30)
parser.add_argument("--patience",   type=int,   default=5)
parser.add_argument("--min-delta",  type=float, default=0.001, dest="min_delta")
parser.add_argument("--lr",         type=float, default=1e-3,  help="LR for the head")
parser.add_argument("--lr-block",   type=float, default=1e-4,  dest="lr_block",
                    help="LR for the last transformer block and norm")
parser.add_argument("--sample-pct", type=float, default=TRAIN_SAMPLE_PCT, dest="sample_pct")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()


def forward_train(vit, x):
    """Frozen prefix with no_grad, then last block + norm + head with gradients."""
    with torch.no_grad():
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)
        for blk in list(vit.blocks)[:-1]:
            x = blk(x)
    x = x.detach()
    x = vit.blocks[-1](x)
    x = vit.norm(x)
    feats  = vit.forward_head(x, pre_logits=True)
    logits = vit.head(feats)
    return logits


def main():
    print(f"Device={DEVICE}  model={MODEL_NAME}")

    train_ds = get_imagenet_data(split="train", sample_pct=args.sample_pct)
    val_ds   = get_imagenet_data(split="val",   sample_pct=VAL_SAMPLE_PCT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

    vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE)
    vit.requires_grad_(False)
    vit.blocks[-1].requires_grad_(True)
    vit.norm.requires_grad_(True)
    vit.head.requires_grad_(True)

    n_block = sum(p.numel() for p in vit.blocks[-1].parameters())
    n_norm  = sum(p.numel() for p in vit.norm.parameters())
    n_head  = sum(p.numel() for p in vit.head.parameters())
    print(f"Trainable params — last block: {n_block:,}  norm: {n_norm:,}  head: {n_head:,}  "
          f"total: {n_block + n_norm + n_head:,}")

    opt = torch.optim.AdamW([
        {"params": list(vit.blocks[-1].parameters()) + list(vit.norm.parameters()),
         "lr": args.lr_block, "weight_decay": 0.05},
        {"params": vit.head.parameters(),
         "lr": args.lr, "weight_decay": 0.0},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    best_val   = float("inf")
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        vit.train()
        train_sum = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {ep}/{args.epochs} [train]"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits = forward_train(vit, x)
                loss   = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            train_sum += loss.item()
        sched.step()
        train_loss = train_sum / len(train_dl)

        vit.eval()
        val_sum = correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {ep}/{args.epochs} [val]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    feats  = vit.forward_features(x)
                    feats  = vit.forward_head(feats, pre_logits=True)
                    logits = vit.head(feats)
                val_sum += F.cross_entropy(logits, y).item()
                correct += (logits.argmax(dim=1) == y).sum().item()
                total   += y.size(0)

        val_loss = val_sum / len(val_dl)
        val_acc  = correct / total
        print(f"Epoch {ep}: train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss; no_improve = 0
            ckpt = {
                "head_state_dict":       vit.head.state_dict(),
                "last_block_state_dict": vit.blocks[-1].state_dict(),
                "norm_state_dict":       vit.norm.state_dict(),
                "epoch":    ep,
                "val_loss": val_loss,
                "val_acc":  val_acc,
                "args":     vars(args),
            }
            torch.save(ckpt, LINEAR_PROBE_LAST_BLOCK_PATH)
            print(f"  → best (val={best_val:.4f}  acc={val_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stopping at epoch {ep}"); break

    print(f"\nDone. Checkpoint saved → {LINEAR_PROBE_LAST_BLOCK_PATH}")


if __name__ == "__main__":
    main()
