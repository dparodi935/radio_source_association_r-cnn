"""Diagnose where the association pipeline is losing accuracy.

Answers three separate questions:
  1. ORACLE  -- is a proposal matching the GT even present? (ceiling)
  2. RANKING -- where does the model rank that proposal? (model quality)
  3. SIGNAL  -- do the cutouts contain visible structure at all? (input quality)

Run:  python diag.py --data-root DATA --split val
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

from dataset import RadioGalaxyDataset, split_mosaics
from model import TinyFastRCNN


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--weights", default=os.path.join(here, "weights.pt"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(a.weights, map_location=device)
    meta = ckpt if isinstance(ckpt, dict) and "state_dict" in ckpt else {}
    size = meta.get("size", 200)
    max_nb = meta.get("max_neighbours", 11)
    ncls = meta.get("num_classes", 3)
    nch = meta.get("in_channels", 1)
    sd = meta.get("state_dict", ckpt)
    print(f"checkpoint: size={size} max_neighbours={max_nb}")

    tr, va, te = split_mosaics(a.data_root, seed=a.seed)
    ids = {"train": tr, "val": va, "test": te}[a.split]
    ds = RadioGalaxyDataset(a.data_root, ids, size=size, max_neighbours=max_nb,
                            verbose=True)

    model = TinyFastRCNN(num_classes=ncls, in_channels=nch).to(device)
    model.load_state_dict(sd)
    model.eval()

    n = min(a.limit, len(ds))
    oracle_hit = 0          # a proposal with IoU>0.9 vs GT exists
    centre_is_truth = 0     # that proposal is #0 (no association)
    model_correct = 0       # model picks a proposal with IoU>0.9
    ranks = []              # rank the model gives the best proposal
    n_props = []
    img_ptp, img_std, img_frac_nonzero = [], [], []
    considered = 0

    for i in range(n):
        image, proposals, gt_boxes, gt_labels = ds[i]
        if len(gt_boxes) == 0:
            continue
        considered += 1
        n_props.append(len(proposals))

        im = image.numpy()
        img_ptp.append(float(np.ptp(im)))
        img_std.append(float(im.std()))
        img_frac_nonzero.append(float((np.abs(im) > 1e-6).mean()))

        ious = box_iou(proposals, gt_boxes)[:, 0]
        best_prop = int(torch.argmax(ious))
        best_iou = float(ious[best_prop])
        if best_iou > 0.9:
            oracle_hit += 1
            if best_prop == 0:
                centre_is_truth += 1

        with torch.no_grad():
            cls_logits, _ = model(image.unsqueeze(0), [proposals])
            scores = F.softmax(cls_logits, dim=1)
            fg = scores[:, 1:].max(dim=1).values

        picked = int(torch.argmax(fg))
        if float(ious[picked]) > 0.9:
            model_correct += 1

        # rank of the oracle-best proposal in the model's ordering (0 = top)
        order = torch.argsort(fg, descending=True)
        rank = int((order == best_prop).nonzero()[0, 0])
        ranks.append(rank)

    if considered == 0:
        print("no samples with GT")
        return

    ranks = np.array(ranks)
    npq = np.array(n_props)
    print(f"\n--- {considered} samples with GT ---")
    print(f"proposals per cutout : median {np.median(npq):.0f}  max {npq.max()}")
    print()
    print("1. ORACLE (is the right answer available?)")
    print(f"   proposal with IoU>0.9 exists : {oracle_hit/considered:.1%}"
          f"   <- your ceiling")
    print(f"   ...and it is 'centre only'   : {centre_is_truth/considered:.1%}"
          f"   <- the no-assoc baseline")
    print()
    print("2. RANKING (can the model find it?)")
    print(f"   model picks a correct box    : {model_correct/considered:.1%}")
    print(f"   rank of the correct proposal : median {np.median(ranks):.0f}, "
          f"mean {ranks.mean():.0f}")
    print(f"   correct in model's top-1     : {(ranks==0).mean():.1%}")
    print(f"   correct in model's top-5     : {(ranks<5).mean():.1%}")
    print(f"   correct in model's top-10%   : {(ranks < npq*0.1).mean():.1%}")
    print(f"   (random guessing would give  : {(1/npq).mean():.2%} top-1)")
    print()
    print("3. SIGNAL (is there anything in the images?)")
    print(f"   pixel range  median {np.median(img_ptp):.3f}")
    print(f"   pixel stddev median {np.median(img_std):.4f}")
    print(f"   frac nonzero median {np.median(img_frac_nonzero):.2%}")
    if np.median(img_std) < 1e-3:
        print("   !! images are near-constant: the network has no input signal")


if __name__ == "__main__":
    main()