import argparse
import json
import sys
from pathlib import Path

_here    = Path(__file__).resolve().parent
_mae_dir = _here.parent
sys.path.insert(0, str(_mae_dir))
sys.path.insert(0, str(_mae_dir.parents[1]))

from _config import MODEL_NAME, MODELS_ROOT, LINEAR_PROBE_LAST_BLOCK_PATH

import torch
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ClientVit, ServerVit
from utils.image import get_imagenet_data
from config import BATCH_SIZE, TRAIN_SAMPLE_PCT, VAL_SAMPLE_PCT

N_BLOCKS = 12  # ViT-Base/16 has 12 transformer blocks

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",        type=int,   default=0)
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--patience",    type=int,   default=8)
parser.add_argument("--min-delta",   type=float, default=0.001, dest="min_delta")
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--sample-pct",  type=float, default=1.0,   dest="sample_pct")
parser.add_argument("--loss",        type=str,   default="ce",  choices=["ce", "kl"])
parser.add_argument("--temperature", type=float, default=4.0,   dest="temperature")
args = parser.parse_args()

T      = args.temperature
DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()


def _kl_loss(student_logits, teacher_probs):
    return F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    teacher_probs, reduction="batchmean") * (T ** 2)


def _load_vit():
    vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE)
    vit.requires_grad_(False)
    _lp = torch.load(LINEAR_PROBE_LAST_BLOCK_PATH, map_location=DEVICE)
    vit.head.load_state_dict(_lp["head_state_dict"])
    vit.blocks[-1].load_state_dict(_lp["last_block_state_dict"])
    vit.norm.load_state_dict(_lp["norm_state_dict"])
    return vit


def _capture_state(modules: list) -> list:
    return [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]


def _restore_state(modules: list, states: list):
    for m, s in zip(modules, states):
        m.load_state_dict(s)


def _train_block(k, vit, vit_teacher, trainable, train_dl, val_dl, log_path):
    client = ClientVit(vit, split_block_number=k + 1, enable_pos_embed=False).to(DEVICE)
    server = ServerVit(vit, split_block_number=k + 1).to(DEVICE).eval()

    trainable_modules = ([vit.patch_embed, vit.blocks[k]] if k == 0
                         else [vit.blocks[k]])
    best_states = _capture_state(trainable_modules)
    if k == 0:
        best_states_extra = vit.cls_token.data.clone()

    opt    = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_val   = float("inf")
    no_improve = 0
    log        = []

    for ep in range(1, args.epochs + 1):
        vit.eval()
        vit.blocks[k].train()
        if k == 0:
            vit.patch_embed.train()

        t_loss = t_correct = t_total = n_batches = 0
        for x, y in tqdm(train_dl, desc=f"  block {k} ep {ep}/{args.epochs} [train]"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            if args.loss == "kl":
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                    teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
                with torch.amp.autocast("cuda", enabled=AMP):
                    logits = vit.head(server(client(x))[:, 0])
                    loss   = _kl_loss(logits, teacher_probs)
            else:
                with torch.amp.autocast("cuda", enabled=AMP):
                    logits = vit.head(server(client(x))[:, 0])
                    loss   = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                t_correct += (logits.argmax(dim=1) == y).sum().item()
            t_loss += loss.item(); t_total += y.size(0); n_batches += 1
        sched.step()

        vit.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  block {k} ep {ep}/{args.epochs} [val  ]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    logits = vit.head(server(client(x))[:, 0])
                    if args.loss == "kl":
                        teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
                        v_loss += _kl_loss(logits, teacher_probs).item()
                    else:
                        v_loss += F.cross_entropy(logits, y).item()
                v_correct += (logits.argmax(dim=1) == y).sum().item()
                v_total   += y.size(0)

        val_loss   = v_loss / len(val_dl)
        val_acc    = v_correct / v_total
        train_loss = t_loss / n_batches
        train_acc  = t_correct / t_total
        print(f"  block {k} ep {ep:>3} │ train {train_loss:.4f} acc {train_acc*100:5.2f}% │ "
              f"val {val_loss:.4f} acc {val_acc*100:5.2f}%")

        log.append({
            "block":      k,
            "epoch":      ep,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
        })
        log_path.write_text(json.dumps(log, indent=2))

        if val_loss < best_val - args.min_delta:
            best_val    = val_loss; no_improve = 0
            best_states = _capture_state(trainable_modules)
            if k == 0:
                best_states_extra = vit.cls_token.data.clone()
            print(f"    → block {k} best (val={best_val:.4f}  acc={val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"    early stopping at epoch {ep}"); break

    _restore_state(trainable_modules, best_states)
    if k == 0:
        vit.cls_token.data.copy_(best_states_extra)

    return val_loss, val_acc


def main():
    save_dir = MODELS_ROOT / "pe_removal_progressive" / "client"
    log_dir  = _here / "training"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    loss_tag = f"_{args.loss}" + (f"_T{T}" if args.loss == "kl" else "")

    print(f"\n{'#'*60}\n  PE-Removal Progressive [{MODEL_NAME}]  loss={args.loss}"
          + (f"  T={T}" if args.loss == "kl" else "") + f"\n{'#'*60}")
    print(f"  Will train all {N_BLOCKS} blocks in sequence (s1..s{N_BLOCKS}).")

    train_ds = get_imagenet_data(split="train", sample_pct=args.sample_pct)
    val_ds   = get_imagenet_data(split="val",   sample_pct=VAL_SAMPLE_PCT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

    vit = _load_vit()

    vit_teacher = None
    if args.loss == "kl":
        vit_teacher = _load_vit()
        vit_teacher.eval()

    for k in range(N_BLOCKS):
        print(f"\n{'='*60}\n  Block {k}/{N_BLOCKS - 1} — "
              + ("patch_embed + cls_token + " if k == 0 else "")
              + f"block_{k}  [{MODEL_NAME}]\n{'='*60}")

        vit.blocks[k].requires_grad_(True)
        if k == 0:
            vit.patch_embed.requires_grad_(True)
            vit.cls_token.requires_grad_(True)
            trainable = (list(vit.patch_embed.parameters()) +
                         [vit.cls_token] +
                         list(vit.blocks[0].parameters()))
        else:
            trainable = list(vit.blocks[k].parameters())

        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")

        log_path = log_dir / f"block{k}{loss_tag}.json"
        val_loss, val_acc = _train_block(k, vit, vit_teacher, trainable,
                                         train_dl, val_dl, log_path)

        vit.blocks[k].requires_grad_(False)
        if k == 0:
            vit.patch_embed.requires_grad_(False)
            vit.cls_token.requires_grad_(False)

        K = k + 1
        client = ClientVit(vit, split_block_number=K, enable_pos_embed=False).to(DEVICE)
        ckpt = {
            "client_state_dict": client.state_dict(),
            "split_block":       K,
            "enable_pos_embed":  False,
            "val_loss":          val_loss,
            "val_acc":           val_acc,
            "args":              vars(args),
        }
        save_path = save_dir / f"s{K}{loss_tag}.pth"
        torch.save(ckpt, save_path)
        print(f"  Saved s{K} → {save_path}")

    if vit_teacher is not None:
        del vit_teacher
    del vit
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
