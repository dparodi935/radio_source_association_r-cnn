import argparse
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from dataset import RadioGalaxyDataset, split_mosaics
from losses import decode_boxes
from model import TinyFastRCNN


CLASS_NAMES = {0: "background", 1: "source"}
CLASS_COLORS = {1: "lime"}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9), aa, ab


def predict_best(model, image, proposals, device, use_regression=False):
    """Returns (best_box, best_score, best_label, best_index).

    Unlike generic detection we want ONE region per cutout: the highest-scoring
    proposal covering the centre component. No NMS -- proposals are a fixed
    lattice of component subsets, not redundant guesses.
    """
    model.eval()
    image = image.to(device)
    proposals = proposals.to(device)

    with torch.no_grad():
        cls_logits, box_deltas = model(image, [proposals])
        scores = F.softmax(cls_logits, dim=1)
        num_classes = scores.size(1)

        fg_scores, fg_labels = scores[:, 1:].max(dim=1)
        fg_labels = fg_labels + 1

        best_idx = int(torch.argmax(fg_scores))
        box = proposals[best_idx]

        if use_regression:
            d = box_deltas.view(-1, num_classes, 4)[best_idx, fg_labels[best_idx]]
            box = decode_boxes(box[None, :], d[None, :])[0]

    return (box.cpu(), float(fg_scores[best_idx]),
            int(fg_labels[best_idx]), best_idx)


def components_in_box(box, centre_xy, neighbour_xy):
    """Which component centres fall inside the predicted region."""
    x1, y1, x2, y2 = [float(v) for v in box]
    inside = [0]                       # the centre component, always
    for i, (x, y) in enumerate(neighbour_xy):
        if x1 <= x <= x2 and y1 <= y <= y2:
            inside.append(i + 1)
    return inside

def count_components(box, neighbour_xy):
    """Number of components inside the predicted region (centre always counts)."""
    if neighbour_xy is None or len(neighbour_xy) == 0:
        return 1
    x1, y1, x2, y2 = [float(v) for v in box]
    xy = np.asarray(neighbour_xy, dtype=float)
    inside = ((xy[:, 0] >= x1) & (xy[:, 0] <= x2) &
              (xy[:, 1] >= y1) & (xy[:, 1] <= y2))
    return 1 + int(inside.sum())


def evaluate(model, dataset, device, use_regression=False, verbose_n=10,
             iou_correct=0.9):
    """Catalogue accuracy vs the no-association baseline, overall and per mosaic.
 
    The GAP (accuracy - baseline) is the figure of merit: a high accuracy means
    nothing if predicting 'never associate' scores the same.
    """ 
    per = defaultdict(lambda: {"n": 0, "correct": 0, "baseline": 0})
    correct = baseline = with_gt = too_many = too_few = 0
    n_multi = multi_correct = 0
 
    for i in range(len(dataset)):
        image, proposals, gt_boxes, gt_labels = dataset[i]
        if len(gt_boxes) == 0:
            continue
        with_gt += 1
        mid = dataset.samples[i]["mosaic_id"]
        per[mid]["n"] += 1
 
        box, score, label, idx = predict_best(
            model, image.unsqueeze(0), proposals, device, use_regression)
 
        gt = [float(v) for v in gt_boxes[0]]
        iou, a_pred, a_gt = _iou([float(v) for v in box], gt)
        is_multi = bool(len(gt_labels) and int(gt_labels[0]) == 2)
        if is_multi:
            n_multi += 1
 
        if iou > iou_correct:
            correct += 1
            per[mid]["correct"] += 1
            if is_multi:
                multi_correct += 1
        elif a_pred > a_gt:
            too_many += 1
        else:
            too_few += 1
 
        b_iou, _, _ = _iou([float(v) for v in proposals[0]], gt)
        if b_iou > iou_correct:
            baseline += 1
            per[mid]["baseline"] += 1
 
        if i < verbose_n:
            print(f"  [{i}] score={score:.3f} prop={idx:4d} IoU={iou:.3f}"
                  f"{'  MC' if is_multi else ''}")
 
    if with_gt == 0:
        print("no samples with ground truth")
        return {}
 
    acc, base = correct / with_gt, baseline / with_gt
    print(f"\ncatalogue accuracy : {acc:.1%}  ({correct}/{with_gt})")
    print(f"no-assoc baseline  : {base:.1%}")
    print(f"GAP                : {acc - base:+.1%}   <- the figure of merit")
    print(f"region too large   : {too_many / with_gt:.1%}")
    print(f"region too small   : {too_few / with_gt:.1%}")
    if n_multi:
        print(f"multi-component    : {multi_correct}/{n_multi} "
              f"= {multi_correct / n_multi:.1%} correct "
              f"({n_multi / with_gt:.1%} of samples)")
 
    print("\nper mosaic:")
    print(f"  {'mosaic':<12}{'n':>6}{'acc':>8}{'base':>8}{'gap':>8}")
    gaps = []
    for mid in sorted(per):
        d = per[mid]
        a, b = d["correct"] / d["n"], d["baseline"] / d["n"]
        gaps.append(a - b)
        print(f"  {mid:<12}{d['n']:>6}{a:>7.1%}{b:>7.1%}{a - b:>+7.1%}")
    gaps = np.array(gaps)
    gap_std = float(gaps.std(ddof=1)) if len(gaps) > 1 else float("nan")
    if len(gaps) > 1:
        print(f"  gap: mean {gaps.mean():+.1%}, sd {gap_std:.1%}, "
              f"min {gaps.min():+.1%}, max {gaps.max():+.1%}")
        if abs(gaps.mean()) < gap_std:
            print("  ** spread exceeds the mean gap: not yet a measurable effect **")
 
    return {
        "n": with_gt, "accuracy": acc, "baseline": base, "gap": acc - base,
        "gap_sd": gap_std, "too_many": too_many / with_gt,
        "too_few": too_few / with_gt, "n_multi": n_multi,
        "multi_acc": (multi_correct / n_multi) if n_multi else float("nan"),
        "n_mosaics": len(per),
    }

def log_result(path, res, meta, split, extra=None):
    """Append one row per evaluated run, so comparisons survive the terminal."""
    if not res:
        return
    row = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "encoding": meta.get("encoding", "radio3"),
        "split": split,
        "size": meta.get("size"),
        "max_neighbours": meta.get("max_neighbours"),
        "in_channels": meta.get("in_channels"),
        "rotations": ",".join(str(r) for r in meta.get("rotations", (0,))),
        "n_train_mosaics": meta.get("n_train_mosaics", ""),
        "epochs": meta.get("epochs", ""),
        "n_samples": res["n"],
        "n_mosaics": res["n_mosaics"],
        "accuracy": round(res["accuracy"], 4),
        "baseline": round(res["baseline"], 4),
        "gap": round(res["gap"], 4),
        "gap_sd": round(res["gap_sd"], 4) if res["gap_sd"] == res["gap_sd"] else "",
        "too_many": round(res["too_many"], 4),
        "too_few": round(res["too_few"], 4),
        "multi_acc": round(res["multi_acc"], 4) if res["multi_acc"] == res["multi_acc"] else "",
    }
    if extra:
        row.update(extra)
 
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"\nlogged -> {path}")


def visualize(image, box, gt_box, score, label, save_path, encoding="radio3", title=None, n_pred=1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return

    from cutouts import ENCODINGS
    
    channels = ENCODINGS[encoding]
    bg_idx = channels.index("sqrt1_30")
    m5_idx = channels.index("mask5")
    
    arr = image.numpy() if torch.is_tensor(image) else np.asarray(image)
    bg = arr[bg_idx]
    mask5 = arr[m5_idx]

    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bg, cmap="inferno", origin="lower", vmin=0, vmax=1)
    if mask5 is not None and mask5.any():
        ax.contour(mask5, levels=[0.5], colors="white",
                   linewidths=0.6, alpha=0.7)

    #add the ground truth box patch
    if gt_box is not None and len(gt_box):
        x1, y1, x2, y2 = [float(v) for v in gt_box[0]]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5,
                                       edgecolor="cyan", facecolor="none",
                                       linestyle="--"))
        
    x1, y1, x2, y2 = [float(v) for v in box]
    color = CLASS_COLORS.get(label, "red")
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5,
                                   edgecolor=color, facecolor="none"))
    ax.text(x1, y1 - 2, f"{n_pred} comp  {score:.2f}",
            color=color, fontsize=8, va="bottom")
    ax.set_title(title or "", fontsize=8)
    ax.axis("off")
    fig.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", default=os.path.join(here, "weights.pt"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=200,
                    help="only used if the checkpoint has no stored config")
    ap.add_argument("--max-neighbours", type=int, default=11,
                    help="only used if the checkpoint has no stored config")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--num-figures", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42,
                    help="must match the seed used in train.py")
    ap.add_argument("--use-regression", action="store_true",
                    help="apply box deltas; off by default (Mostert disables it)")
    ap.add_argument("--output-dir", default=os.path.join(here, "inference_outputs"))
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load checkpoint first: it dictates size/max_neighbours ---
    ckpt = torch.load(a.weights, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        size = ckpt.get("size", a.size)
        max_nb = ckpt.get("max_neighbours", a.max_neighbours)
        num_classes = ckpt.get("num_classes", a.num_classes)
        in_channels = ckpt.get("in_channels", 3)
        encoding = ckpt.get("encoding", "radio3")
        print(f"checkpoint config: size={size} max_neighbours={max_nb} "
              f"num_classes={num_classes} in_channels={in_channels}")
    else:
        state_dict = ckpt
        size, max_nb = a.size, a.max_neighbours
        num_classes, in_channels = a.num_classes, 3
        encoding = "radio3"
        print(f"WARNING: legacy checkpoint with no stored config; "
              f"falling back to --size {size} --max-neighbours {max_nb}. "
              f"These MUST match what training used.")

    train_ids, val_ids, test_ids = split_mosaics(a.data_root, seed=a.seed)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[a.split]
    if not ids:
        raise SystemExit(f"'{a.split}' split is empty: {train_ids} {val_ids} {test_ids}")
    print(f"{a.split} mosaics: {ids}")

    ds = RadioGalaxyDataset(a.data_root, ids, size=size, max_neighbours=max_nb,
                            encoding=encoding)

    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)

    res = evaluate(model, ds, device, use_regression=a.use_regression)
    log_result(os.path.join(here, "results.csv"), res, ckpt, a.split)
    
    os.makedirs(a.output_dir, exist_ok=True)
    
    # pick interesting samples rather than the first N
    order = sorted(range(len(ds)),
                   key=lambda i: (len(ds.samples[i]["gt_label"]) == 0,
                                  ds.samples[i]["gt_label"][0] != 2
                                  if len(ds.samples[i]["gt_label"]) else True))
    
    for n, i in enumerate(order[:a.num_figures]):
        image, proposals, gt_boxes, gt_labels = ds[i]
        box, score, label, _ = predict_best(
            model, image.unsqueeze(0), proposals, device, a.use_regression)

        nb_xy = ds.samples[i].get("neighbour_xy")
        n_pred = count_components(box, nb_xy)
        n_true = count_components(gt_boxes[0], nb_xy) if len(gt_boxes) else 0

        visualize(image, box, gt_boxes, score, label,
                  os.path.join(a.output_dir, f"sample_{n:04d}.png"),
                  encoding,
                  title=f"{ds.samples[i]['source_name']}\n"
                        f"pred {n_pred} comp / true {n_true}",
                  n_pred=n_pred)
    print(f"figures -> {a.output_dir}/")
    
    
if __name__ == "__main__":
    main()
