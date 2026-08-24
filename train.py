import argparse
import os
from torchvision.ops import box_iou
import torch
from torch.utils.data import DataLoader

from losses import listwise_loss
from dataset import RadioGalaxyDataset, split_mosaics, collate_fn
from cutouts import ENCODINGS, n_channels
from model import TinyFastRCNN


def run_epoch(model, loader, optimizer, device):
    """One pass. optimizer=None -> evaluation (no grad, no update).

    Loss is listwise: one softmax per cutout over its own proposals, which is
    the objective the catalogue-accuracy metric actually measures.
    fg_iou/bg_iou are unused now (kept so the call sites don't change).
    
    Returns: 
        average classification loss, 
        average regression loss, 
        number correct, 
        number scored (when highest scored proposal is not the best one), 
        0
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

            # use listwise loss, normalising over every proposal
            # this is instead of Mostert's method, which scored each proposal separately
            cls_loss = listwise_loss(cls_logits, proposals, gt_boxes)
            reg_loss = box_deltas.sum() * 0.0      # box regression disabled
            loss = cls_loss

        if train:
            optimizer.zero_grad() # clears gradients from prev step 
            loss.backward() # goes backward through the NN and computes the gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step() # changes weights using calculated gradient

        # diagnostics: is the top-scored proposal the correct one?
        with torch.no_grad():
            start = 0
            for props, gt in zip(proposals, gt_boxes):
                end = start + len(props)
                if len(gt) > 0:
                    ious = box_iou(props, gt)[:, 0]
                    target = int(torch.argmax(ious)) # id of prop with largest overlap
                    if float(ious[target]) >= 0.9:
                        scores = cls_logits[start:end, 1:].max(dim=1).values 
                        
                        # adds 1 if the highest scored box is the one with the highest overlap with the ground truth
                        n_correct += int(int(torch.argmax(scores)) == target)
                         
                        n_scored += 1
                start = end

        tot_cls += float(cls_loss.detach())
        tot_reg += float(reg_loss.detach())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return tot_cls / n_batches, tot_reg / n_batches, n_correct, n_scored, 0


def train(data_root, out_path, num_classes=2, size=200,
          max_neighbours=8, batch_size=4, num_epochs=20, lr=1e-4,
          max_train=None, max_val=None,
          rotations=(0, 25, 50, 100), encoding="radio3",
          seed=42, device=None, torch_seed=0):

    in_channels = n_channels(encoding) 
    print(f"encoding: {encoding} -> {in_channels} channels "
          f"{ENCODINGS[encoding]}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ids, val_ids, test_ids = split_mosaics(
        data_root, seed=seed, max_train=max_train, max_val=max_val)
    print(f"train mosaics: {train_ids}")
    print(f"val mosaics:   {val_ids}")
    print(f"test mosaics:  {test_ids}  (held out, not loaded here)")

    # load training dataset
    train_ds = RadioGalaxyDataset(data_root, train_ids, size=size,
                                  max_neighbours=max_neighbours,
                                  rotations=rotations, encoding=encoding)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn)
    
    
    # load validation dataset
    val_dl = None
    if val_ids:  # check to see there are mosaics available for validation
        val_ds = RadioGalaxyDataset(data_root, val_ids, size=size,
                                max_neighbours=max_neighbours,
                                rotations=(0,), encoding=encoding)        
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)

    torch.manual_seed(torch_seed)
    
    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    
    # Use Adam algorithm for optimiser (thing that changes the weights)
    # Uses adapative learning rates for individual parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    # config travels with the weights so infer.py can't use a mismatched size
    meta = {"size": size, "max_neighbours": max_neighbours,
            "num_classes": num_classes, "in_channels": in_channels,
            "encoding": encoding, "rotations": tuple(rotations),
            "n_train_mosaics": len(train_ids), "epochs": num_epochs}

    best_acc = -1.0
    for epoch in range(num_epochs):
        # run training epoch 
        c, r, pos, neg, _ = run_epoch(
            model, train_dl, optimizer, device)
        
        line = (f"epoch {epoch+1}/{num_epochs}  train loss {c:.4f} "
                f"top1 {pos/max(neg,1):.1%} ({pos}/{neg})")

        # tests model on validation dataset, but does not update weights from this
        # this is to guard against overtraining
        if val_dl is not None:
            vc, vr, vpos, vneg, _ = run_epoch(model, val_dl, None, device)
            
            vacc = vpos / max(vneg, 1)
            line += f"  |  val loss {vc:.4f} top1 {vacc:.1%}"
            
            # select on the metric we care about (validation accuracy), not the loss
            if vacc > best_acc:
                best_acc = vacc
                torch.save({"state_dict": model.state_dict(), **meta}, out_path) # saves current best version of the model
                line += "  *saved"
        print(line)

    # if there are no mosaics for validation, simply save the latest state of the model
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
    ap.add_argument("--encoding", default="radio3", choices=sorted(ENCODINGS))
    ap.add_argument("--max-neighbours", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap on number of training mosaics")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--rotations", type=int, nargs="+", default=[0, 25, 50, 100])
    ap.add_argument("--torch-seed", type=int, default=0)
    a = ap.parse_args()

    train(a.data_root, a.out, num_classes=a.num_classes, size=a.size,
          max_neighbours=a.max_neighbours, batch_size=a.batch_size,
          num_epochs=a.epochs, lr=a.lr, fg_iou=a.fg_iou, bg_iou=a.bg_iou,
          max_train=a.max_train, max_val=a.max_val, encoding=a.encoding, 
          rotations=tuple(a.rotations), torch_seed=a.torch_seed)
