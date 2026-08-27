import argparse
import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from dataset import RadioGalaxyDataset, split_mosaics
from model import TinyFastRCNN
from cutouts import DIR_INFER_OUTPUTS

CLASS_NAMES = {0: "background", 1: "source"}
CLASS_COLORS = {1: "lime"}


def predict_best(model, image, proposals, device):
    """Returns (best_box, best_score, best_index).

    Unlike generic detection we want ONE region per cutout: the highest-scoring
    proposal covering the centre component. No NMS -- proposals are a fixed
    lattice of component subsets, not redundant guesses.
    """
    model.eval()
    image = image.to(device)
    proposals = proposals.to(device)

    with torch.no_grad():
        # run model on image. Generates raw scores for each proposal
        cls_logits, box_deltas = model(image, [proposals])
        scores = F.softmax(cls_logits, dim=1)

        fg_scores, fg_labels = scores[:, 1:].max(dim=1)
        
        # choose the highest score and select the corresponding box
        best_idx = int(torch.argmax(fg_scores))
        box = proposals[best_idx]

    return (box.cpu(), float(fg_scores[best_idx]), best_idx)


def get_components_in_box(box, neighbour_xy):
    """Returns a set of component indices inside the box based on spatial coordinates. 
    Index 0 represents the central component."""
    
    # The central component is always the origin/target, so it's always included
    components = {0} 
    
    if neighbour_xy is not None and len(neighbour_xy) > 0:
        x1, y1, x2, y2 = [float(v) for v in box]
        xy = np.asarray(neighbour_xy, dtype=float)
        
        # Find which neighbours physically fall inside the box bounds
        inside = ((xy[:, 0] >= x1) & (xy[:, 0] <= x2) &
                  (xy[:, 1] >= y1) & (xy[:, 1] <= y2))
        
        # Add the indices of the neighbours that are inside (starting at 1)
        for i, is_in in enumerate(inside, start=1):
            if is_in:
                components.add(i)
                
    return components


from scipy.stats import chi2

def gap_heterogeneity(per):
    """Q test: is the per-mosaic gap spread more than sampling noise?"""
    gaps, vars_, ns = [], [], []
    all_d = []
    for mid in sorted(per):
        cf = np.asarray(per[mid]["cf"], int)
        bf = np.asarray(per[mid]["bf"], int)
        d = cf - bf                                  # -1, 0, +1 per cutout
        all_d.append(d)
        gaps.append(d.mean())
        vars_.append(d.var(ddof=1) / len(d))
        ns.append(len(d))

    gaps, vars_ = np.array(gaps), np.array(vars_)
    d_all = np.concatenate(all_d)
    gap_pooled = d_all.mean()
    se_pooled = np.sqrt(d_all.var(ddof=1) / len(d_all))

    k = len(gaps)
    Q = float(((gaps - gap_pooled) ** 2 / vars_).sum())
    crit = chi2.ppf(0.95, k - 1)

    print(f"\npooled gap: {gap_pooled:+.1%} +/- {se_pooled:.1%} (paired SE)")
    print(f"  95% CI: [{gap_pooled - 1.96*se_pooled:+.1%}, "
          f"{gap_pooled + 1.96*se_pooled:+.1%}]")
    print(f"heterogeneity Q = {Q:.2f}, df = {k-1}, "
          f"chi2 95th pct = {crit:.2f} -> "
          f"{'mosaics genuinely differ' if Q > crit else 'consistent with one common gap'}")
    return gap_pooled, se_pooled, Q


def evaluate(model, dataset, device, verbose_n=10):
    """Catalogue accuracy vs the no-association baseline, overall and per mosaic.

    The GAP (accuracy - baseline) is the figure of merit: a high accuracy means
    nothing if predicting 'never associate' scores the same.
    """ 
    
    per = defaultdict(lambda: {"n": 0, "correct": 0, "baseline": 0,
                               "cf": [], "bf": []})       # flag lists

    correct = baseline = with_gt = too_many = too_few = 0
    n_multi = multi_correct = 0

    for i in range(len(dataset)):
        # retrieve the image, the proposals and ground truth and insert those into the model to return a chosen box and its score 
        image, proposals, gt_boxes, gt_labels = dataset[i]
        if len(gt_boxes) == 0:
            continue
            
        gt_idx_list = dataset.samples[i].get("gt_idx")
        if gt_idx_list is None:
            raise ValueError(f"Sample {i} missing 'gt_idx'. Update cutouts.py to save 'gt_idx' and regenerate the dataset.")
            
        with_gt += 1
        mid = dataset.samples[i]["mosaic_id"]
        per[mid]["n"] += 1

        box, score, idx = predict_best(model, image.unsqueeze(0), proposals, device)

        # GET PREDICTED COMPONENTS (SPATIAL OVERLAP)
        nb_xy = dataset.samples[i].get("neighbour_xy")
        pred_set = get_components_in_box(box, nb_xy)
        
        # GET TRUE COMPONENTS (EXACT CATALOGUE IDS)
        true_set = set(gt_idx_list)
        
        baseline_set = {0} # Baseline simply predicts the central component by itself

        is_multi = len(true_set) > 1
        is_correct = (pred_set == true_set)
        is_base = (baseline_set == true_set)
        
        per[mid]["cf"].append(is_correct)
        per[mid]["bf"].append(is_base)
                
        if is_multi:
            n_multi += 1
        
        if is_correct:
            correct += 1
            per[mid]["correct"] += 1
            if is_multi:
                multi_correct += 1
        elif true_set < pred_set:
            too_many += 1
        elif pred_set < true_set:
            too_few += 1
        else:
            too_many += 1        # mixed case, following Mostert
        
        if is_base:
            baseline += 1
            per[mid]["baseline"] += 1

        if i < verbose_n:
            print(f"  [{i}] score={score:.3f} prop={idx:4d} "
                  f"Pred:{pred_set} GT:{true_set}"
                  f"{'  MC' if is_multi else ''}")
        
    if with_gt == 0:
        print("no samples with ground truth")
        return {}
    
    # CALCULATE AND PRINT STATISTICS
    acc, base = correct / with_gt, baseline / with_gt
    print(f"\ncatalogue accuracy : {acc:.1%}  ({correct}/{with_gt})")
    print(f"no-assoc baseline  : {base:.1%}")
    print(f"GAP                : {acc - base:+.1%}   <- the figure of merit")
    print(f"Too many components   : {too_many / with_gt:.1%}")
    print(f"Too few components   : {too_few / with_gt:.1%}")
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
    
    gap_heterogeneity(per)
    
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

        
def visualize(image, box, gt_box, score, save_path, encoding="radio3", title=None, n_pred=1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return

    from cutouts import ENCODINGS
    
    channels = ENCODINGS[encoding]
    
    bg_idx = channels.index("sqrt1_30") if "sqrt1_30" in channels else 0
    m5_idx = channels.index("mask5") if "mask5" in channels else None
    w1_idx = channels.index("w1") if "w1" in channels else None
    if w1_idx is None:
         w1_idx = channels.index("w1masked") if "w1masked" in channels else None
         
    arr = image.numpy() if torch.is_tensor(image) else np.asarray(image)
    
    panels_spec = [(bg_idx, "inferno", "Radio")]
    if w1_idx is not None:
        panels_spec.append((w1_idx, "viridis", "Optical (W1)"))

    ncols = len(panels_spec)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5.5))
    axes = [axes] if ncols == 1 else list(axes)

    for ax, (idx, cmap, label) in zip(axes, panels_spec):
        ax.imshow(arr[idx], cmap=cmap, origin="lower", vmin=0, vmax=1)
        
        if m5_idx is not None:
            mask5 = arr[m5_idx]
            if mask5 is not None and mask5.any():
                ax.contour(mask5, levels=[0.5], colors="white",
                           linewidths=0.6, alpha=0.7)

        if gt_box is not None and len(gt_box):
            x1, y1, x2, y2 = [float(v) for v in gt_box[0]]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5,
                                           edgecolor="cyan", facecolor="none",
                                           linestyle="--"))
            
        x1, y1, x2, y2 = [float(v) for v in box]
        color = "lime"
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=1.5,
                                       edgecolor=color, facecolor="none"))
        
        if ax is axes[0]:
            ax.text(x1, y1 - 2, f"{n_pred} comp  {score:.2f}",
                    color=color, fontsize=8, va="bottom")
            
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        
    if title:
        fig.suptitle(title, fontsize=12)
        
    fig.tight_layout()
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
    ap.add_argument("--output-dir", default=DIR_INFER_OUTPUTS)
    a = ap.parse_args()
    
    output_dir = os.path.join(a.data_root, a.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # DETERMINISTICALLY SPLIT IDS
    train_ids, val_ids, test_ids = split_mosaics(a.data_root, seed=a.seed)
    
    test_ids = ["164_47", "170_49"]
    train_ids, val_ids = [], []

    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[a.split]
    if not ids:
        raise SystemExit(f"'{a.split}' split is empty: {train_ids} {val_ids} {test_ids}")
    print(f"{a.split} mosaics: {ids}")


    # LOAD DATASET AND MODEL
    ds = RadioGalaxyDataset(a.data_root, ids, size=size, max_neighbours=max_nb,
                            encoding=encoding)

    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)

    
    # RUN THE MODEL ON LOADED DATASET AND LOG RESULTS
    res = evaluate(model, ds, device)
    log_result(os.path.join(here, "results.csv"), res, ckpt, a.split)
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    # VISUALISE SELECTION OF OUTPUTS
    order = sorted(range(len(ds)),
                   key=lambda i: (len(ds.samples[i].get("gt_label", [])) == 0,
                                  ds.samples[i].get("gt_label", [0])[0] != 2
                                  if len(ds.samples[i].get("gt_label", [])) else True))
    
    for n, i in enumerate(order[:a.num_figures]):
        image, proposals, gt_boxes, gt_labels = ds[i]
        box, score, _ = predict_best(model, image.unsqueeze(0), proposals, device)

        nb_xy = ds.samples[i].get("neighbour_xy")
        
        # Calculate predicted components spatially, but get true components exactly from ID
        n_pred = len(get_components_in_box(box, nb_xy))
        gt_idx_list = ds.samples[i].get("gt_idx", [0])
        n_true = len(gt_idx_list)
                
        infer_output_dir = os.path.join(output_dir,encoding)
        os.makedirs(infer_output_dir, exist_ok=True)

        
        visualize(image, box, gt_boxes, score,
                  os.path.join(infer_output_dir, f"sample_{n:04d}.png"),
                  encoding,
                  title=f"{ds.samples[i]['source_name']}\n"
                        f"pred {n_pred} comp / true {n_true}",
                  n_pred=n_pred)
    print(f"figures -> {infer_output_dir}/")
    
    
if __name__ == "__main__":
    main()