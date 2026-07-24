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
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.split_vit import ClientVit, ServerVit
from models.position_predictor import ContextualPositionPredictor
from utils.image import get_imagenet_data
from config import BATCH_SIZE, VAL_SAMPLE_PCT

N_PATCH = 196
N_POS   = 197

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cuda",           type=int,   default=0)
parser.add_argument("--split",          type=int,   nargs="+", default=list(range(1, 13)),
                    help="1-based split points to train")
# ── mode flags (mutually exclusive) ──
_mode = parser.add_mutually_exclusive_group()
_mode.add_argument("--adv-only",        action="store_true", dest="adv_only",
                   help="adversarial training with PE kept (no PE removal)")
_mode.add_argument("--pe-removal-only", action="store_true", dest="pe_removal_only",
                   help="progressive PE-removal fine-tuning only — no adversarial term, no TPP")
# ── Adversarial adversarial hyperparams ──
parser.add_argument("--rounds",         type=int,   default=10,
                    help="alternating min-max + convergence rounds per block")
parser.add_argument("--epochs",         type=int,   default=5,
                    help="epochs for the alternating min-max phase per round")
parser.add_argument("--tpp-epochs",     type=int,   default=10,  dest="tpp_epochs",
                    help="max epochs for TPP convergence phase")
parser.add_argument("--tpp-patience",   type=int,   default=3,   dest="tpp_patience",
                    help="early-stop patience for TPP convergence")
parser.add_argument("--tpp-target-acc", type=float, default=0.80, dest="tpp_target_acc",
                    help="stop TPP convergence early once val accuracy reaches this threshold")
parser.add_argument("--tpp-sample-pct", type=float, default=0.1,  dest="tpp_sample_pct",
                    help="train-set fraction for TPP convergence phases")
parser.add_argument("--lam-max",        type=float, default=0.1,  dest="lam_max",
                    help="max weight on -CE(TPP) in client loss")
parser.add_argument("--warmup-frac",    type=float, default=0.5,  dest="warmup_frac",
                    help="fraction of round-1 epochs spent ramping lambda 0→lam_max")
parser.add_argument("--ce-cap",         type=float, default=5.28, dest="ce_cap",
                    help="per-token clamp on CE(TPP) in client loss (default: ln(197), chance level)")
parser.add_argument("--block-converge",    action="store_true", dest="block_converge",
                    help="run a KL fine-tuning phase after all adversarial rounds")
parser.add_argument("--block-conv-epochs", type=int, default=10, dest="block_conv_epochs",
                    help="epochs for the block convergence phase (requires --block-converge)")
# ── PE-only hyperparams ──
parser.add_argument("--pe-only-epochs",   type=int,   default=15,  dest="pe_only_epochs",
                    help="epochs per block for --pe-removal-only mode")
parser.add_argument("--pe-only-patience", type=int,   default=8,   dest="pe_only_patience",
                    help="early-stop patience for --pe-removal-only mode")
# ── shared ──
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--sample-pct",     type=float, default=0.1,  dest="sample_pct",
                    help="train-set fraction")
parser.add_argument("--temperature",    type=float, default=4.0,
                    help="KL distillation temperature")
parser.add_argument("--min-delta",      type=float, default=0.001, dest="min_delta")
parser.add_argument("--resume-from",    type=str,   default=None,  dest="resume_from",
                    help="path to a client checkpoint (.pth) to merge into the ViT before training "
                         "(e.g. s8_full.pth to resume PE-only from an Adversarial-trained base)")

args = parser.parse_args()

T      = args.temperature
DEVICE = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
AMP    = torch.cuda.is_available()

if args.pe_removal_only:
    ENABLE_PE   = False
    FILE_TAG    = "_pe_only"
    LOG_SUBDIR  = "pe_only"
elif args.adv_only:
    ENABLE_PE   = True
    FILE_TAG    = "_adv_only"
    LOG_SUBDIR  = "adv_only"
else:
    ENABLE_PE   = False
    FILE_TAG    = "_full"
    LOG_SUBDIR  = "full"


# ─── model loading ───────────────────────────────────────────────────────────

def _load_vit():
    vit = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000).to(DEVICE)
    vit.requires_grad_(False)
    _lp = torch.load(LINEAR_PROBE_LAST_BLOCK_PATH, map_location=DEVICE)
    vit.head.load_state_dict(_lp["head_state_dict"])
    vit.blocks[-1].load_state_dict(_lp["last_block_state_dict"])
    vit.norm.load_state_dict(_lp["norm_state_dict"])
    return vit


# ─── loss helpers ────────────────────────────────────────────────────────────

def _kl_loss(student_logits, teacher_probs):
    return F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    teacher_probs, reduction="batchmean") * (T ** 2)


def _lambda_for_epoch(ep, total_epochs, warmup_frac, lam_max):
    warmup_epochs = max(1, int(total_epochs * warmup_frac))
    return lam_max if ep >= warmup_epochs else lam_max * ep / warmup_epochs


def _make_tpp_batch(smashed):
    B = smashed.shape[0]
    patch_perm = torch.stack([torch.randperm(N_PATCH, device=DEVICE) + 1 for _ in range(B)])
    perm   = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=DEVICE), patch_perm], dim=1)
    bidx   = torch.arange(B, device=DEVICE).unsqueeze(1)
    gt_pos = torch.arange(N_POS, device=DEVICE).unsqueeze(0).expand(B, -1)
    return smashed[bidx, perm][:, 1:], gt_pos[bidx, perm][:, 1:]


def _get_trainable(vit, K):
    if K == 1:
        vit.patch_embed.requires_grad_(True)
        vit.cls_token.requires_grad_(True)
        vit.blocks[0].requires_grad_(True)
        params = (list(vit.patch_embed.parameters()) + [vit.cls_token] +
                  list(vit.blocks[0].parameters()))
        return params, [vit.patch_embed, vit.blocks[0]]
    vit.blocks[K - 1].requires_grad_(True)
    return list(vit.blocks[K - 1].parameters()), [vit.blocks[K - 1]]


def _freeze_split(vit, K):
    if K == 1:
        vit.patch_embed.requires_grad_(False)
        vit.cls_token.requires_grad_(False)
        vit.blocks[0].requires_grad_(False)
    else:
        vit.blocks[K - 1].requires_grad_(False)


# ─── training phases ─────────────────────────────────────────────────────────

def _train_pe_only_block(name, vit, vit_teacher, trainable_params, trainable_modules,
                         K, train_dl, val_dl, log_path):
    client = ClientVit(vit, split_block_number=K, enable_pos_embed=False).to(DEVICE)
    server = ServerVit(vit, split_block_number=K).to(DEVICE).eval()

    opt    = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.pe_only_epochs)

    best_val       = float("inf")
    best_val_acc   = 0.0
    no_improve     = 0
    best_states    = [{k: v.clone() for k, v in m.state_dict().items()} for m in trainable_modules]
    best_cls_state = vit.cls_token.data.clone() if K == 1 else None
    log = []

    for ep in range(1, args.pe_only_epochs + 1):
        vit.eval()
        for m in trainable_modules:
            m.train()

        t_loss = t_correct = t_total = n_batches = 0
        for x, y in tqdm(train_dl, desc=f"  [{name}] ep {ep}/{args.pe_only_epochs} [train]"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits = vit.head(server(client(x))[:, 0])
                loss   = _kl_loss(logits, teacher_probs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                t_correct += (logits.argmax(dim=1) == y).sum().item()
            t_loss += loss.item(); t_total += y.size(0); n_batches += 1
        sched.step()

        vit.eval()
        v_loss = v_correct = v_total = v_batches = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  [{name}] ep {ep}/{args.pe_only_epochs} [val  ]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
                    logits = vit.head(server(client(x))[:, 0])
                    v_loss += _kl_loss(logits, teacher_probs).item()
                v_correct += (logits.argmax(dim=1) == y).sum().item()
                v_total += y.size(0); v_batches += 1

        val_loss = v_loss / v_batches
        val_acc  = v_correct / v_total
        print(f"  [{name}] ep {ep:>3} │ train {t_loss/n_batches:.4f} acc {t_correct/t_total*100:5.2f}% │ "
              f"val {val_loss:.4f} acc {val_acc*100:5.2f}%")

        log.append({
            "stage": name, "epoch": ep,
            "train_loss": round(t_loss / n_batches, 6),
            "train_acc":  round(t_correct / t_total, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
        })
        log_path.write_text(json.dumps(log, indent=2))

        if val_loss < best_val - args.min_delta:
            best_val    = val_loss; best_val_acc = val_acc; no_improve = 0
            best_states = [{k: v.clone() for k, v in m.state_dict().items()} for m in trainable_modules]
            if K == 1:
                best_cls_state = vit.cls_token.data.clone()
            print(f"    → best (val={best_val:.4f}  acc={val_acc*100:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= args.pe_only_patience:
                print(f"    early stopping at epoch {ep}"); break

    for m, s in zip(trainable_modules, best_states):
        m.load_state_dict(s)
    if K == 1:
        vit.cls_token.data.copy_(best_cls_state)
    return best_val, best_val_acc


def _finetune_block(name, vit, vit_teacher, trainable_params, trainable_modules,
                    K, train_dl, val_dl, log_path):
    client = ClientVit(vit, split_block_number=K, enable_pos_embed=ENABLE_PE).to(DEVICE)
    server = ServerVit(vit, split_block_number=K).to(DEVICE).eval()

    opt    = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.block_conv_epochs)

    val_loss = val_acc = 0.0
    log = []

    for ep in range(1, args.block_conv_epochs + 1):
        vit.eval()
        for m in trainable_modules:
            m.train()

        t_loss = t_correct = t_total = n_batches = 0
        for x, y in tqdm(train_dl, desc=f"  [{name}] ep {ep}/{args.block_conv_epochs} [train]"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits = vit.head(server(client(x))[:, 0])
                loss   = _kl_loss(logits, teacher_probs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                t_correct += (logits.argmax(dim=1) == y).sum().item()
            t_loss += loss.item(); t_total += y.size(0); n_batches += 1
        sched.step()

        vit.eval()
        v_loss = v_correct = v_total = v_batches = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  [{name}] ep {ep}/{args.block_conv_epochs} [val  ]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
                    logits = vit.head(server(client(x))[:, 0])
                    v_loss += _kl_loss(logits, teacher_probs).item()
                v_correct += (logits.argmax(dim=1) == y).sum().item()
                v_total += y.size(0); v_batches += 1

        val_loss = v_loss / v_batches
        val_acc  = v_correct / v_total
        print(f"  [{name}] ep {ep:>3} │ train {t_loss/n_batches:.4f} acc {t_correct/t_total*100:5.2f}% │ "
              f"val {val_loss:.4f} acc {val_acc*100:5.2f}%")

        log.append({
            "stage": name, "epoch": ep,
            "train_loss": round(t_loss / n_batches, 6),
            "train_acc":  round(t_correct / t_total, 6),
            "val_loss":   round(val_loss, 6),
            "val_acc":    round(val_acc, 6),
        })
        log_path.write_text(json.dumps(log, indent=2))

    return val_loss, val_acc


def _train_round(name, vit, vit_teacher, tpp,
                 trainable_params, trainable_modules,
                 K, train_dl, val_dl, warmup, log_path):
    client = ClientVit(vit, split_block_number=K, enable_pos_embed=ENABLE_PE).to(DEVICE)
    server = ServerVit(vit, split_block_number=K).to(DEVICE).eval()

    opt_client = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    opt_tpp    = torch.optim.Adam(tpp.parameters(), lr=args.lr)
    scaler_c   = torch.amp.GradScaler("cuda", enabled=AMP)
    scaler_t   = torch.amp.GradScaler("cuda", enabled=AMP)
    sched_c    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_client, T_max=args.epochs)

    val_util = val_acc = val_tpp_acc = 0.0
    log = []

    for ep in range(1, args.epochs + 1):
        lam = _lambda_for_epoch(ep, args.epochs, args.warmup_frac, args.lam_max) if warmup else args.lam_max

        vit.eval()
        for m in trainable_modules:
            m.train()
        tpp.train()

        t_util = t_correct = t_total = t_tpp_loss = t_tpp_correct = t_tpp_total = n_batches = 0
        for x, y in tqdm(train_dl, desc=f"  [{name}] ep {ep}/{args.epochs} [train] lam={lam:.3f}"):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            # ---- TPP step: minimize CE(TPP(z.detach()), true_pos) ----
            for p in tpp.parameters():
                p.requires_grad_(True)
            opt_tpp.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                smashed = client(x)
            inp, target = _make_tpp_batch(smashed)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits_tpp = tpp(inp)
                loss_tpp   = F.cross_entropy(logits_tpp.reshape(-1, N_POS), target.reshape(-1))
            scaler_t.scale(loss_tpp).backward()
            scaler_t.step(opt_tpp); scaler_t.update()
            with torch.no_grad():
                t_tpp_correct += (logits_tpp.argmax(dim=-1) == target).sum().item()
            t_tpp_loss += loss_tpp.item(); t_tpp_total += target.numel()

            # ---- client step: min KL – λ * CE_capped(TPP(z), true_pos) ----
            for p in tpp.parameters():
                p.requires_grad_(False)
            opt_client.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                teacher_probs = F.softmax(vit_teacher(x) / T, dim=-1)
            with torch.amp.autocast("cuda", enabled=AMP):
                z              = client(x)
                student_logits = vit.head(server(z)[:, 0])
                loss_util      = _kl_loss(student_logits, teacher_probs)
                inp2, target2  = _make_tpp_batch(z)
                loss_pos_capped = torch.clamp(
                    F.cross_entropy(tpp(inp2).reshape(-1, N_POS),
                                    target2.reshape(-1), reduction="none"),
                    max=args.ce_cap).mean()
                loss_client = loss_util - lam * loss_pos_capped
            scaler_c.scale(loss_client).backward()
            scaler_c.unscale_(opt_client)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler_c.step(opt_client); scaler_c.update()
            with torch.no_grad():
                t_correct += (student_logits.argmax(dim=1) == y).sum().item()
            t_util += loss_util.item(); t_total += y.size(0); n_batches += 1
        sched_c.step()

        vit.eval(); tpp.eval()
        v_util = v_correct = v_total = v_tpp_loss = v_tpp_correct = v_tpp_total = v_batches = 0
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"  [{name}] ep {ep}/{args.epochs} [val  ]"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    z              = client(x)
                    student_logits = vit.head(server(z)[:, 0])
                    teacher_probs  = F.softmax(vit_teacher(x) / T, dim=-1)
                    v_util        += _kl_loss(student_logits, teacher_probs).item()
                    inp, target    = _make_tpp_batch(z)
                    logits_tpp     = tpp(inp)
                    v_tpp_loss    += F.cross_entropy(logits_tpp.reshape(-1, N_POS),
                                                     target.reshape(-1)).item()
                v_correct     += (student_logits.argmax(dim=1) == y).sum().item()
                v_tpp_correct += (logits_tpp.argmax(dim=-1) == target).sum().item()
                v_total += y.size(0); v_tpp_total += target.numel(); v_batches += 1

        val_util     = v_util / v_batches
        val_acc      = v_correct / v_total
        val_tpp_loss = v_tpp_loss / v_batches
        val_tpp_acc  = v_tpp_correct / v_tpp_total
        print(f"  [{name}] ep {ep:>3} │ lam={lam:.3f} │ "
              f"util={val_util:.4f} acc={val_acc*100:5.2f}% │ "
              f"tpp loss={val_tpp_loss:.4f} acc={val_tpp_acc*100:5.2f}%")

        log.append({
            "stage": name, "epoch": ep, "lambda": round(lam, 4),
            "train_util_loss": round(t_util / n_batches, 6),
            "train_acc":       round(t_correct / t_total, 6),
            "train_tpp_loss":  round(t_tpp_loss / n_batches, 6),
            "train_tpp_acc":   round(t_tpp_correct / t_tpp_total, 6),
            "val_util_loss":   round(val_util, 6),
            "val_acc":         round(val_acc, 6),
            "val_tpp_loss":    round(val_tpp_loss, 6),
            "val_tpp_acc":     round(val_tpp_acc, 6),
        })
        log_path.write_text(json.dumps(log, indent=2))

    return tpp, val_util, val_acc, val_tpp_acc


def _converge_tpp(name, vit, tpp, K, train_dl, val_dl, log_path):
    client = ClientVit(vit, split_block_number=K, enable_pos_embed=ENABLE_PE).to(DEVICE)
    for p in tpp.parameters():
        p.requires_grad_(True)
    opt    = torch.optim.Adam(tpp.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)

    best_val, best_state, no_improve = float("inf"), None, 0
    log = []

    for ep in range(1, args.tpp_epochs + 1):
        tpp.train()
        t_loss = t_correct = t_total = n_batches = 0
        for x, _ in tqdm(train_dl, desc=f"  [{name}] ep {ep}/{args.tpp_epochs} [tpp  ]"):
            x = x.to(DEVICE, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=AMP):
                smashed = client(x)
            inp, target = _make_tpp_batch(smashed)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP):
                logits = tpp(inp)
                loss   = F.cross_entropy(logits.reshape(-1, N_POS), target.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                t_correct += (logits.argmax(dim=-1) == target).sum().item()
            t_loss += loss.item(); t_total += target.numel(); n_batches += 1

        tpp.eval()
        v_loss = v_correct = v_total = v_batches = 0
        with torch.no_grad():
            for x, _ in tqdm(val_dl, desc=f"  [{name}] ep {ep}/{args.tpp_epochs} [val  ]"):
                x = x.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=AMP):
                    smashed = client(x)
                    inp, target = _make_tpp_batch(smashed)
                    logits = tpp(inp)
                    v_loss += F.cross_entropy(logits.reshape(-1, N_POS), target.reshape(-1)).item()
                v_correct += (logits.argmax(dim=-1) == target).sum().item()
                v_total += target.numel(); v_batches += 1

        val_loss = v_loss / v_batches
        val_acc  = v_correct / v_total
        print(f"  [{name}] ep {ep:>3} │ tpp loss={val_loss:.4f} acc={val_acc*100:5.2f}%")

        log.append({
            "stage": name, "epoch": ep,
            "train_tpp_loss": round(t_loss / n_batches, 6),
            "train_tpp_acc":  round(t_correct / t_total, 6),
            "val_tpp_loss":   round(val_loss, 6),
            "val_tpp_acc":    round(val_acc, 6),
        })
        log_path.write_text(json.dumps(log, indent=2))

        if val_loss < best_val - args.min_delta:
            best_val = val_loss; no_improve = 0
            best_state = {k: v.clone() for k, v in tpp.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.tpp_patience:
                print(f"    TPP converged (early stop at epoch {ep})"); break

        if val_acc >= args.tpp_target_acc:
            best_state = {k: v.clone() for k, v in tpp.state_dict().items()}
            print(f"    TPP reached target {val_acc*100:.2f}% — stopping early"); break

    if best_state is not None:
        tpp.load_state_dict(best_state)
    tpp.eval()
    return tpp


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    client_dir = MODELS_ROOT / "adversarial" / "client"
    tpp_dir    = MODELS_ROOT / "adversarial" / "tpp"
    client_dir.mkdir(parents=True, exist_ok=True)
    tpp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = get_imagenet_data(split="train", sample_pct=args.sample_pct)
    val_ds   = get_imagenet_data(split="val",   sample_pct=VAL_SAMPLE_PCT)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=8, pin_memory=True)

    if not args.pe_removal_only:
        tpp_train_ds = get_imagenet_data(split="train", sample_pct=args.tpp_sample_pct)
        tpp_train_dl = DataLoader(tpp_train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=8, pin_memory=True, drop_last=True)

    vit = _load_vit()
    vit.requires_grad_(False)
    if args.resume_from is not None:
        _ckpt = torch.load(args.resume_from, map_location=DEVICE)
        _sd   = vit.state_dict()
        _sd.update(_ckpt["client_state_dict"])
        vit.load_state_dict(_sd)
        print(f"  Resumed ViT weights from {args.resume_from}")
    vit_teacher = _load_vit()
    vit_teacher.requires_grad_(False).eval()

    if args.pe_removal_only:
        mode = "PE-removal only (no adversarial)"
    elif args.adv_only:
        mode = "adversarial only (PE kept)"
    else:
        mode = "adversarial + PE removal"

    for K in sorted(args.split):
        log_dir = _here / "training" / f"s{K}" / LOG_SUBDIR
        log_dir.mkdir(parents=True, exist_ok=True)

        trainable_params, trainable_modules = _get_trainable(vit, K)

        if args.pe_removal_only:
            print(f"\n{'#'*60}\n  Adversarial [{MODEL_NAME}]  split={K}  ({mode})"
                  f"\n  epochs={args.pe_only_epochs}  patience={args.pe_only_patience}\n{'#'*60}")

            val_loss, val_acc = _train_pe_only_block(
                f"s{K}_pe_only", vit, vit_teacher,
                trainable_params, trainable_modules,
                K, train_dl, val_dl,
                log_path=log_dir / "pe_only.json")

            client = ClientVit(vit, split_block_number=K, enable_pos_embed=False).to(DEVICE)
            ckpt = {
                "client_state_dict": client.state_dict(),
                "split_block":       K,
                "enable_pos_embed":  False,
                "val_loss":          val_loss,
                "val_acc":           val_acc,
                "args":              vars(args),
            }
            client_path = client_dir / f"s{K}{FILE_TAG}.pth"
            torch.save(ckpt, client_path)
            print(f"  Saved client → {client_path}")

            _freeze_split(vit, K)
            del client

        else:
            print(f"\n{'#'*60}\n  Adversarial [{MODEL_NAME}]  split={K}  ({mode})"
                  f"\n  lam_max={args.lam_max}  rounds={args.rounds}  epochs/round={args.epochs}\n{'#'*60}")

            tpp = ContextualPositionPredictor().to(DEVICE)

            val_util = val_acc = val_tpp_acc = None
            for r in range(1, args.rounds + 1):
                print(f"\n{'='*60}\n  Round {r}/{args.rounds}\n{'='*60}")

                tpp, val_util, val_acc, val_tpp_acc = _train_round(
                    f"round{r}", vit, vit_teacher, tpp,
                    trainable_params, trainable_modules,
                    K, train_dl, val_dl,
                    warmup=(r == 1),
                    log_path=log_dir / f"round{r}.json")

                tpp = _converge_tpp(
                    f"round{r}_converge", vit, tpp, K,
                    tpp_train_dl, val_dl,
                    log_path=log_dir / f"round{r}_converge.json")

            if args.block_converge:
                print(f"\n{'='*60}\n  Block convergence (KL fine-tuning)\n{'='*60}")
                val_util, val_acc = _finetune_block(
                    "block_converge", vit, vit_teacher,
                    trainable_params, trainable_modules,
                    K, train_dl, val_dl,
                    log_path=log_dir / "block_converge.json")

            client = ClientVit(vit, split_block_number=K, enable_pos_embed=ENABLE_PE).to(DEVICE)
            ckpt = {
                "client_state_dict": client.state_dict(),
                "split_block":       K,
                "enable_pos_embed":  ENABLE_PE,
                "val_loss":          val_util,
                "val_acc":           val_acc,
                "val_tpp_acc":       val_tpp_acc,
                "args":              vars(args),
            }
            client_path = client_dir / f"s{K}{FILE_TAG}.pth"
            torch.save(ckpt, client_path)
            print(f"  Saved client → {client_path}")

            tpp_ckpt = {"state_dict": tpp.state_dict(), "val_tpp_acc": val_tpp_acc, "args": vars(args)}
            torch.save(tpp_ckpt, tpp_dir / f"s{K}{FILE_TAG}.pth")

            _freeze_split(vit, K)
            del tpp, client

        torch.cuda.empty_cache()

    del vit, vit_teacher
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
