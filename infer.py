import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import RadioGalaxyDataset, split_mosaics
from losses import decode_boxes
from model import TinyFastRCNN

CLASS_NAMES = {0: "background", 1: "single", 2: "multi-component"}
CLASS_COLORS = {1: "lime", 2: "orange"}


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


def evaluate(model, dataset, device, use_regression=False, verbose_n=10):
    """Catalogue accuracy (Mostert et al. 2022): a prediction is correct only if
    the predicted region encloses exactly the right set of components.

    Also reports the no-association baseline: assume every component stands alone.
    """
    n = len(dataset)
    correct = baseline = with_gt = 0
    too_many = too_few = 0

    for i in range(n):
        image, proposals, gt_boxes, gt_labels = dataset[i]
        if len(gt_boxes) == 0:
            continue
        with_gt += 1

        box, score, label, idx = predict_best(
            model, image.unsqueeze(0), proposals, device, use_regression)

        # IoU of the predicted region against the true region
        gt = gt_boxes[0]
        ix1 = max(float(box[0]), float(gt[0])); iy1 = max(float(box[1]), float(gt[1]))
        ix2 = min(float(box[2]), float(gt[2])); iy2 = min(float(box[3]), float(gt[3]))
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
        a2 = (float(gt[2]) - float(gt[0])) * (float(gt[3]) - float(gt[1]))
        iou = inter / max(a1 + a2 - inter, 1e-9)

        if iou > 0.9:
            correct += 1
        elif a1 > a2:
            too_many += 1
        else:
            too_few += 1

        # baseline: predict "no association" (proposal 0 == centre alone)
        base = proposals[0]
        bx1 = max(float(base[0]), float(gt[0])); by1 = max(float(base[1]), float(gt[1]))
        bx2 = min(float(base[2]), float(gt[2])); by2 = min(float(base[3]), float(gt[3]))
        binter = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        ba = (float(base[2]) - float(base[0])) * (float(base[3]) - float(base[1]))
        if binter / max(ba + a2 - binter, 1e-9) > 0.9:
            baseline += 1

        if i < verbose_n:
            print(f"  [{i}] {CLASS_NAMES[label]:>15s} score={score:.3f} "
                  f"prop={idx:4d} IoU={iou:.3f}")

    if with_gt == 0:
        print("no samples with ground truth")
        return {}

    res = {
        "n": with_gt,
        "accuracy": correct / with_gt,
        "baseline": baseline / with_gt,
        "too_many": too_many / with_gt,
        "too_few": too_few / with_gt,
    }
    print(f"\ncatalogue accuracy : {res['accuracy']:.1%}  ({correct}/{with_gt})")
    print(f"no-assoc baseline  : {res['baseline']:.1%}")
    print(f"region too large   : {res['too_many']:.1%}")
    print(f"region too small   : {res['too_few']:.1%}")
    return res


def visualize(image, box, gt_box, score, label, save_path, title=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return

    arr = image.numpy() if torch.is_tensor(image) else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):      # (C,H,W)
        bg = arr[0]
        mask5 = arr[2] if arr.shape[0] == 3 else None
    else:                                              # legacy (H,W)
        bg = np.squeeze(arr)
        mask5 = None

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
    ax.text(x1, y1 - 2, f"{CLASS_NAMES.get(label, label)} {score:.2f}",
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
    ap.add_argument("--num-classes", type=int, default=3)
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
        in_channels = ckpt.get("in_channels", 1)
        print(f"checkpoint config: size={size} max_neighbours={max_nb} "
              f"num_classes={num_classes} in_channels={in_channels}")
    else:
        state_dict = ckpt
        size, max_nb = a.size, a.max_neighbours
        num_classes, in_channels = a.num_classes, 1
        print(f"WARNING: legacy checkpoint with no stored config; "
              f"falling back to --size {size} --max-neighbours {max_nb}. "
              f"These MUST match what training used.")

    train_ids, val_ids, test_ids = split_mosaics(a.data_root, seed=a.seed)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[a.split]
    if not ids:
        raise SystemExit(f"'{a.split}' split is empty: {train_ids} {val_ids} {test_ids}")
    print(f"{a.split} mosaics: {ids}")

    ds = RadioGalaxyDataset(a.data_root, ids, size=size, max_neighbours=max_nb)

    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(state_dict)

    res = evaluate(model, ds, device, use_regression=a.use_regression)

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
        visualize(image, box, gt_boxes, score, label,
                  os.path.join(a.output_dir, f"sample_{i:04d}.png"),
                  title=ds.samples[i]["source_name"])
    print(f"figures -> {a.output_dir}/")
    
    
if __name__ == "__main__":
    main()
