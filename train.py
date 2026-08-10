import argparse
import os
from losses import listwise_loss
from torchvision.ops import box_iou

import torch
from torch.utils.data import DataLoader

from dataset import RadioGalaxyDataset, split_mosaics, collate_fn
from losses import compute_losses, match_boxes_to_gt
from model import TinyFastRCNN

def balance(labels, bg_per_fg=3, generator=None):
    """Keep all fg, a limited sample of bg, drop ignores. Returns a bool mask."""
    fg = labels > 0
    bg = labels == 0
    n_keep = max(int(fg.sum()) * bg_per_fg, 32)
    bg_idx = torch.nonzero(bg, as_tuple=True)[0]
    if bg_idx.numel() > n_keep:
        perm = torch.randperm(bg_idx.numel(), device=labels.device, generator=generator)
        bg_idx = bg_idx[perm[:n_keep]]
    mask = fg.clone()
    mask[bg_idx] = True
    return mask


def run_epoch(model, loader, optimizer, num_classes, fg_iou, bg_iou, device):
    """One pass. optimizer=None -> evaluation (no grad, no update).

    Loss is listwise: one softmax per cutout over its own proposals, which is
    the objective the catalogue-accuracy metric actually measures.
    fg_iou/bg_iou are unused now (kept so the call sites don't change).
    """
    train = optimizer is not None
    model.train(train)

    tot_cls = tot_reg = 0.0
    n_correct = n_scored = 0
    n_batches = 0

    for images, proposals, gt_boxes, gt_labels in loader:
        images = images.to(device)
        proposals = [p.to(device) for p in proposals]
        gt_boxes = [g.to(device) for g in gt_boxes]

        with torch.set_grad_enabled(train):
            cls_logits, box_deltas = model(images, proposals)

            cls_loss = listwise_loss(cls_logits, proposals, gt_boxes)
            reg_loss = box_deltas.sum() * 0.0      # box regression disabled
            loss = cls_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # diagnostics: is the top-scored proposal the correct one?
        with torch.no_grad():
            start = 0
            for props, gt in zip(proposals, gt_boxes):
                end = start + len(props)
                if len(gt) > 0:
                    ious = box_iou(props, gt)[:, 0]
                    tgt = int(torch.argmax(ious))
                    if float(ious[tgt]) >= 0.9:
                        sc = cls_logits[start:end, 1:].max(dim=1).values
                        n_correct += int(int(torch.argmax(sc)) == tgt)
                        n_scored += 1
                start = end

        tot_cls += float(cls_loss.detach())
        tot_reg += float(reg_loss.detach())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return tot_cls / n_batches, tot_reg / n_batches, n_correct, n_scored, 0


def train(data_root, out_path, num_classes=3, in_channels=3, size=200,
          max_neighbours=11, batch_size=4, num_epochs=20, lr=1e-4,
          fg_iou=0.8, bg_iou=0.5, max_train=None, max_val=None,
          seed=42, device=None):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ids, val_ids, test_ids = split_mosaics(
        data_root, seed=seed, max_train=max_train, max_val=max_val)
    print(f"train mosaics: {train_ids}")
    print(f"val mosaics:   {val_ids}")
    print(f"test mosaics:  {test_ids}  (held out, not loaded here)")

    train_ds = RadioGalaxyDataset(data_root, train_ids, size=size,
                                  max_neighbours=max_neighbours)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)

    val_dl = None
    if val_ids:
        val_ds = RadioGalaxyDataset(data_root, val_ids, size=size,
                                    max_neighbours=max_neighbours)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # config travels with the weights so infer.py can't use a mismatched size
    meta = {"size": size, "max_neighbours": max_neighbours,
            "num_classes": num_classes, "in_channels": in_channels}

    best_acc = -1.0
    for epoch in range(num_epochs):
        c, r, pos, neg, _ = run_epoch(
            model, train_dl, optimizer, num_classes, fg_iou, bg_iou, device)
        line = (f"epoch {epoch+1}/{num_epochs}  train loss {c:.4f} "
                f"top1 {pos/max(neg,1):.1%} ({pos}/{neg})")

        if val_dl is not None:
            vc, vr, vpos, vneg, _ = run_epoch(model, val_dl, None, num_classes,
                                              fg_iou, bg_iou, device)
            vacc = vpos / max(vneg, 1)
            line += f"  |  val loss {vc:.4f} top1 {vacc:.1%}"
            # select on the metric we care about, not the loss
            if vacc > best_acc:
                best_acc = vacc
                torch.save({"state_dict": model.state_dict(), **meta}, out_path)
                line += "  *saved"
        print(line)

    if val_dl is None:
        torch.save({"state_dict": model.state_dict(), **meta}, out_path)
    print(f"weights -> {out_path}")
    if best_acc >= 0:
        print(f"best val top1: {best_acc:.1%}")
    return model


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default=os.path.join(here, "weights.pt"))
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--max-neighbours", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fg-iou", type=float, default=0.8)
    ap.add_argument("--bg-iou", type=float, default=0.5)
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap on number of training mosaics")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--num-classes", type=int, default=3)
    a = ap.parse_args()

    train(a.data_root, a.out, num_classes=a.num_classes, size=a.size,
          max_neighbours=a.max_neighbours, batch_size=a.batch_size,
          in_channels=a.in_channels,
          num_epochs=a.epochs, lr=a.lr, fg_iou=a.fg_iou, bg_iou=a.bg_iou,
          max_train=a.max_train, max_val=a.max_val)
