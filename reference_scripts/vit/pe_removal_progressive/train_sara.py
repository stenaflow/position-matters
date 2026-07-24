import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ClientVit
from models.position_predictor import ContextualPositionPredictor
from models.mae import TokenToImageDecoder
from models.decoder import AttackerDecoder
from utils.image import get_imagenet_data
from config import SPLIT_BLOCK, BATCH_SIZE, TRAIN_SAMPLE_PCT, VAL_SAMPLE_PCT

_project_root = Path(__file__).resolve().parents[2]
_client_root  = _project_root / "saved_models" / "pe_removal_progressive" / "client"
_sara_root    = _project_root / "saved_models" / "pe_removal_progressive" / "sara"

N_PATCH       = 196
REDUCTION_MIN = 0.0
REDUCTION_MAX = 0.95

parser = argparse.ArgumentParser()
parser.add_argument("--cuda",         type=int,   default=0)
parser.add_argument("--split",        type=int,   nargs="+", default=[SPLIT_BLOCK], metavar="K")
parser.add_argument("--sample-pct",   type=float, default=TRAIN_SAMPLE_PCT, dest="sample_pct")
parser.add_argument("--tpp-epochs",   type=int,   default=20, dest="tpp_epochs")
parser.add_argument("--tpp-patience", type=int,   default=5,  dest="tpp_patience")
parser.add_argument("--mae-epochs",   type=int,   default=10, dest="mae_epochs")
parser.add_argument("--mae-patience", type=int,   default=3,  dest="mae_patience")
parser.add_argument("--dec-epochs",   type=int,   default=20, dest="dec_epochs")
parser.add_argument("--dec-patience", type=int,   default=5,  dest="dec_patience")
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--loss",         type=str,   default="kl", choices=["ce", "kl"],
                    help="client checkpoint loss tag (default: kl)")
parser.add_argument("--temperature",  type=float, default=4.0,
                    help="distillation temperature used when training the client (default: 4.0)")
parser.add_argument("--skip-tpp",     action="store_true", dest="skip_tpp")
parser.add_argument("--skip-mae",     action="store_true", dest="skip_mae")
parser.add_argument("--decoder-only", action="store_true", dest="decoder_only")
args = parser.parse_args()

DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()


def random_mask(smashed: torch.Tensor) -> torch.Tensor:
    B, N, D = smashed.shape
    n_reduce = int((N - 1) * (REDUCTION_MIN + torch.rand(1).item() * (REDUCTION_MAX - REDUCTION_MIN)))
    if n_reduce == 0:
        return smashed.clone()
    masked = smashed.clone()
    for b in range(B):
        idx = torch.randperm(N - 1, device=smashed.device)[:n_reduce] + 1
        masked[b, idx] = 0.0
    return masked


def load_client(K):
    tag       = f"s{K}_{args.loss}" + (f"_T{args.temperature}" if args.loss == "kl" else "")
    ckpt_path = _client_root / f"{tag}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PE-removal progressive client not found: {ckpt_path}\n"
                                f"Run train_client.py --split {K} --loss {args.loss}"
                                + (f" --temperature {args.temperature}" if args.loss == "kl" else "")
                                + " first.")
    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    vit    = timm.create_model("vit_base_patch16_224", pretrained=True)
    client = ClientVit(vit, K, enable_pos_embed=False)
    client.load_state_dict(ckpt["client_state_dict"])
    return client.to(DEVICE).eval()


def _train_tpp(client, train_dl, val_dl, save_path):
    print(f"\n{'='*60}\n[1/3] ContextualPositionPredictor\n{'='*60}")
    tpp    = ContextualPositionPredictor().to(DEVICE)
    opt    = torch.optim.Adam(tpp.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.tpp_epochs)
    best_val, no_improve = float("inf"), 0

    for ep in range(1, args.tpp_epochs + 1):
        tpp.train()
        t_loss = t_correct = t_tokens = n_batches = 0
        for x, _ in tqdm(train_dl, desc=f"  TPP {ep}/{args.tpp_epochs} [train]"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                smashed = client(x)
            B = smashed.shape[0]
            patch_perm = torch.stack([torch.randperm(N_PATCH, device=DEVICE) + 1 for _ in range(B)])
            perm   = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=DEVICE), patch_perm], dim=1)
            bidx   = torch.arange(B, device=DEVICE).unsqueeze(1)
            gt_pos = torch.arange(197, device=DEVICE).unsqueeze(0).expand(B, -1)
            target = gt_pos[bidx, perm][:, 1:]
            inp    = smashed[bidx, perm][:, 1:]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits = tpp(inp)
                loss   = F.cross_entropy(logits.reshape(-1, 197), target.reshape(-1))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                t_correct += (logits.argmax(dim=-1) == target).sum().item()
            t_loss += loss.item(); t_tokens += target.numel(); n_batches += 1
        sched.step()

        tpp.eval()
        v_loss = v_correct = v_tokens = v_batches = 0
        with torch.no_grad():
            for x, _ in tqdm(val_dl, desc=f"  TPP {ep}/{args.tpp_epochs} [val  ]"):
                x = x.to(DEVICE, non_blocking=True)
                smashed = client(x); B = smashed.shape[0]
                pp = torch.stack([torch.randperm(N_PATCH, device=DEVICE) + 1 for _ in range(B)])
                pm = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=DEVICE), pp], dim=1)
                bi = torch.arange(B, device=DEVICE).unsqueeze(1)
                gt = torch.arange(197, device=DEVICE).unsqueeze(0).expand(B, -1)
                tgt = gt[bi, pm][:, 1:]; inp = smashed[bi, pm][:, 1:]
                with torch.amp.autocast("cuda", enabled=AMP):
                    lg = tpp(inp); loss = F.cross_entropy(lg.reshape(-1, 197), tgt.reshape(-1))
                v_loss += loss.item(); v_correct += (lg.argmax(dim=-1) == tgt).sum().item()
                v_tokens += tgt.numel(); v_batches += 1

        val_loss = v_loss / v_batches
        print(f"  TPP ep {ep:>3} │ train {t_loss/n_batches:.4f} acc {t_correct/t_tokens*100:5.2f}% │ "
              f"val {val_loss:.4f} acc {v_correct/v_tokens*100:5.2f}%")
        ckpt = {"state_dict": tpp.state_dict(), "epoch": ep, "val_loss": val_loss, "args": vars(args)}
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0; torch.save(ckpt, save_path)
        else:
            no_improve += 1
            if no_improve >= args.tpp_patience:
                print(f"    early stopping at epoch {ep}"); break

    _ckpt = torch.load(save_path, map_location=DEVICE)
    tpp.load_state_dict(_ckpt.get("state_dict", _ckpt)); tpp.eval()
    return tpp


def _train_mae(client, train_dl, val_dl, split_block, save_path):
    print(f"\n{'='*60}\n[2/3] TokenToImageDecoder (MAE)\n{'='*60}")
    mae = TokenToImageDecoder().to(DEVICE)
    criterion = nn.MSELoss()
    opt    = torch.optim.AdamW(mae.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.mae_epochs)
    best_val, no_improve = float("inf"), 0

    for ep in range(1, args.mae_epochs + 1):
        mae.train(); t_loss = n_batches = 0
        for x, _ in tqdm(train_dl, desc=f"  MAE {ep}/{args.mae_epochs} [train]"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                smashed = client(x); masked = random_mask(smashed)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                loss = criterion(mae(masked[:, 1:]), smashed[:, 1:])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            t_loss += loss.item(); n_batches += 1
        sched.step()

        mae.eval(); v_loss = v_count = 0
        with torch.no_grad():
            for x, _ in tqdm(val_dl, desc=f"  MAE {ep}/{args.mae_epochs} [val  ]"):
                x = x.to(DEVICE, non_blocking=True)
                smashed = client(x); masked = random_mask(smashed)
                with torch.amp.autocast("cuda", enabled=AMP):
                    v_loss += criterion(mae(masked[:, 1:]), smashed[:, 1:]).item()
                v_count += 1
        val_loss = v_loss / v_count
        print(f"  MAE ep {ep:>3} │ train {t_loss/n_batches:.6f}  val {val_loss:.6f}")

        ckpt = {"decoder_state_dict": mae.state_dict(), "split_block": split_block,
                "epoch": ep, "val_loss": val_loss, "args": vars(args)}
        if val_loss < best_val:
            best_val = val_loss; no_improve = 0; torch.save(ckpt, save_path)
        else:
            no_improve += 1
            if no_improve >= args.mae_patience:
                print(f"    early stopping at epoch {ep}"); break

    mae.load_state_dict(torch.load(save_path, map_location=DEVICE)["decoder_state_dict"]); mae.eval()
    return mae


def _train_decoder(client, mae, train_dl, val_dl, split_block, save_path):
    print(f"\n{'='*60}\n[3/3] AttackerDecoder\n{'='*60}")
    decoder   = AttackerDecoder().to(DEVICE)
    criterion = nn.MSELoss()
    opt    = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.dec_epochs)
    best_val, no_improve = float("inf"), 0

    for ep in range(1, args.dec_epochs + 1):
        decoder.train(); t_loss = n_batches = 0
        for x, _ in tqdm(train_dl, desc=f"  Dec {ep}/{args.dec_epochs} [train]"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad():
                smashed = client(x); masked = random_mask(smashed)
                tokens  = mae(masked[:, 1:])
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                loss = criterion(decoder(tokens), x)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            t_loss += loss.item(); n_batches += 1
        sched.step()

        decoder.eval(); v_loss = v_count = 0
        with torch.no_grad():
            for x, _ in tqdm(val_dl, desc=f"  Dec {ep}/{args.dec_epochs} [val  ]"):
                x = x.to(DEVICE, non_blocking=True)
                smashed = client(x); masked = random_mask(smashed)
                tokens  = mae(masked[:, 1:])
                with torch.amp.autocast("cuda", enabled=AMP):
                    v_loss += criterion(decoder(tokens), x).item()
                v_count += 1
        val_loss = v_loss / v_count
        print(f"  Dec ep {ep:>3} │ train {t_loss/n_batches:.4f}  val {val_loss:.4f}")

        ckpt = {"decoder_state_dict": decoder.state_dict(), "split_block": split_block,
                "epoch": ep, "val_loss": val_loss, "args": vars(args)}
        if val_loss < best_val - 1e-3:
            best_val = val_loss; no_improve = 0; torch.save(ckpt, save_path)
        else:
            no_improve += 1
            if no_improve >= args.dec_patience:
                print(f"    early stopping at epoch {ep}"); break

    decoder.load_state_dict(torch.load(save_path, map_location=DEVICE)["decoder_state_dict"])
    decoder.eval()
    return decoder


def main():
    train_ds = get_imagenet_data(split="train", sample_pct=args.sample_pct)
    val_ds   = get_imagenet_data(split="val",   sample_pct=VAL_SAMPLE_PCT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

    for K in args.split:
        print(f"\n{'#'*60}\n  PE-Removal Progressive SARA  split={K}  loss={args.loss}"
              + (f"  T={args.temperature}" if args.loss == "kl" else "") + f"\n{'#'*60}")

        client = load_client(K)
        for p in client.parameters():
            p.requires_grad_(False)

        tag      = f"s{K}_{args.loss}" + (f"_T{args.temperature}" if args.loss == "kl" else "")
        tpp_path = _sara_root / "tpp"     / f"{tag}.pth"
        mae_path = _sara_root / "mae"     / f"{tag}.pth"
        dec_path = _sara_root / "decoder" / f"{tag}.pth"
        for p in (tpp_path, mae_path, dec_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        if args.decoder_only or args.skip_tpp:
            tpp = ContextualPositionPredictor().to(DEVICE)
            _ckpt = torch.load(tpp_path, map_location=DEVICE)
            tpp.load_state_dict(_ckpt.get("state_dict", _ckpt)); tpp.eval()
            print("[skip] Loaded existing TPP")
        else:
            tpp = _train_tpp(client, train_dl, val_dl, tpp_path)

        if args.decoder_only or args.skip_mae:
            mae = TokenToImageDecoder().to(DEVICE)
            mae.load_state_dict(torch.load(mae_path, map_location=DEVICE)["decoder_state_dict"])
            mae.eval()
            print("[skip] Loaded existing MAE")
        else:
            mae = _train_mae(client, train_dl, val_dl, K, mae_path)

        _train_decoder(client, mae, train_dl, val_dl, K, dec_path)

        del client, tpp, mae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
