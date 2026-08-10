import torch
import torch.nn.functional as F
from torchvision.ops import box_iou


def encode_boxes(proposals, gt_boxes):
    """
    Converts absolute [x1,y1,x2,y2] ground-truth boxes into the (dx,dy,dw,dh)
    delta parameterization relative to each proposal box, which is what the
    reg_head is trained to predict.

    Args:
        proposals: (N, 4) [x1,y1,x2,y2]
        gt_boxes:  (N, 4) [x1,y1,x2,y2], already matched 1-to-1 with proposals

    Returns:
        deltas: (N, 4) [dx, dy, dw, dh]
    """
    px1, py1, px2, py2 = proposals.unbind(dim=1)
    gx1, gy1, gx2, gy2 = gt_boxes.unbind(dim=1)

    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    pcx = px1 + 0.5 * pw
    pcy = py1 + 0.5 * ph

    gw = (gx2 - gx1).clamp(min=1e-6)
    gh = (gy2 - gy1).clamp(min=1e-6)
    gcx = gx1 + 0.5 * gw
    gcy = gy1 + 0.5 * gh

    dx = (gcx - pcx) / pw
    dy = (gcy - pcy) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(proposals, deltas):
    """
    Inverse of encode_boxes: turns predicted (dx,dy,dw,dh) deltas back into
    absolute [x1,y1,x2,y2] boxes. Used at inference time.

    Args:
        proposals: (N, 4) [x1,y1,x2,y2]
        deltas:    (N, 4) [dx, dy, dw, dh]

    Returns:
        boxes: (N, 4) [x1,y1,x2,y2]
    """
    px1, py1, px2, py2 = proposals.unbind(dim=1)
    pw = (px2 - px1).clamp(min=1e-6)
    ph = (py2 - py1).clamp(min=1e-6)
    pcx = px1 + 0.5 * pw
    pcy = py1 + 0.5 * ph

    dx, dy, dw, dh = deltas.unbind(dim=1)
    gcx = dx * pw + pcx
    gcy = dy * ph + pcy
    gw = torch.exp(dw) * pw
    gh = torch.exp(dh) * ph

    x1 = gcx - 0.5 * gw
    y1 = gcy - 0.5 * gh
    x2 = gcx + 0.5 * gw
    y2 = gcy + 0.5 * gh

    return torch.stack([x1, y1, x2, y2], dim=1)

def listwise_loss(cls_logits, proposals, gt_boxes, iou_thresh=0.9):
    """One softmax per cutout over ITS OWN proposals.

    cls_logits: (total_proposals, C) concatenated across the batch, in the same
                order RoIAlign produced them (image 0's boxes, then image 1's...).
    proposals:  list of (n_i, 4) tensors, one per image
    gt_boxes:   list of (0|1, 4) tensors, one per image
    """
    losses = []
    start = 0
    for props, gt in zip(proposals, gt_boxes):
        end = start + len(props)
        if len(gt) > 0:
            ious = box_iou(props, gt)[:, 0]
            target = int(torch.argmax(ious))
            if float(ious[target]) >= iou_thresh:
                score = cls_logits[start:end, 1:].max(dim=1).values  # (n_i,)
                losses.append(F.cross_entropy(
                    score[None, :],
                    torch.tensor([target], device=score.device)))
        start = end
    if not losses:
        return cls_logits.sum() * 0.0
    return torch.stack(losses).mean()

def match_boxes_to_gt(proposals, gt_boxes, gt_labels,
                      iou_threshold=0.5, bg_threshold=None):
    """See below. If bg_threshold is given, proposals with IoU in
    [bg_threshold, iou_threshold) are marked IGNORE (-1) and excluded from the
    loss, following Mostert et al. 2022 (bg < 0.5, fg > 0.8)."""
    return _match(proposals, gt_boxes, gt_labels, iou_threshold, bg_threshold)


def _match(proposals, gt_boxes, gt_labels, iou_threshold=0.5, bg_threshold=None):
    """
    Assigns each proposal box (from PyBDSF, for ONE image) a class label and a
    matched ground-truth box, based on IoU overlap.

    Args:
        proposals: (N, 4) candidate boxes for one image
        gt_boxes:  (M, 4) ground-truth boxes for that same image
        gt_labels: (M,)   ground-truth class id per box (1..num_classes-1)
        iou_threshold: proposals with best IoU below this are labeled background (0)

    Returns:
        matched_labels:   (N,) LongTensor, 0 = background
        matched_gt_boxes: (N, 4) the ground-truth box each proposal was matched to
                           (meaningless for background rows, but kept for shape consistency)
    """
    if gt_boxes.numel() == 0:
        # No ground truth in this image -> everything is background
        matched_labels = torch.zeros(proposals.size(0), dtype=torch.long, device=proposals.device)
        matched_gt_boxes = torch.zeros_like(proposals)
        return matched_labels, matched_gt_boxes
    #print("ground truth present")
    ious = box_iou(proposals, gt_boxes)          # (N, M)
    best_iou, best_gt_idx = ious.max(dim=1)       # (N,), (N,)
        
    matched_labels = gt_labels[best_gt_idx].clone()
    if bg_threshold is None:
        matched_labels[best_iou < iou_threshold] = 0        # background
    else:
        ignore = (best_iou >= bg_threshold) & (best_iou < iou_threshold)
        matched_labels[best_iou < bg_threshold] = 0         # background
        matched_labels[ignore] = -1                         # excluded from loss

    matched_gt_boxes = gt_boxes[best_gt_idx]

    return matched_labels, matched_gt_boxes


def compute_losses(cls_logits, box_deltas, matched_labels, matched_proposals,
                    matched_gt_boxes, num_classes):
    """
    Computes classification + class-specific box-regression loss for one batch
    (already concatenated across all images).
    Returns loss from differn

    Args:
        cls_logits:        (N, num_classes) - Raw score
        box_deltas:        (N, num_classes * 4) - Raw box delta
        matched_labels:    (N,) LongTensor, 0 = background #GT
        matched_proposals: (N, 4) the original proposal box for each row
        matched_gt_boxes:  (N, 4) the ground-truth box each proposal was matched to #GT
        num_classes:       total number of classes including background

    Returns:
        cls_loss: scalar tensor
        reg_loss: scalar tensor (0 if there are no positive/foreground boxes)
    """
    # Compares predicted probability to actual label
    # cross_entropy takes in the three scores for each label, applies a softmax (restricting range to [0,1])
    # then uses -log(probability) for the true label score () 
    # Drop IGNORE rows (label -1) entirely: they contribute to neither loss.
    keep = matched_labels >= 0
    if keep.sum() == 0:
        zero = cls_logits.sum() * 0.0
        return zero, box_deltas.sum() * 0.0
    cls_logits = cls_logits[keep]
    box_deltas = box_deltas[keep]
    matched_labels = matched_labels[keep]
    matched_proposals = matched_proposals[keep]
    matched_gt_boxes = matched_gt_boxes[keep]

    cls_loss = F.cross_entropy(cls_logits, matched_labels)

    pos_mask = matched_labels > 0 #mask to filter out background
    #if there are no galaxies (only background), set the loss to zero
    if pos_mask.sum() == 0:
        reg_loss = box_deltas.sum() * 0.0  # zero, but keeps it in the graph safely
        return cls_loss, reg_loss

    pos_labels = matched_labels[pos_mask]
    pos_proposals = matched_proposals[pos_mask]
    pos_gt_boxes = matched_gt_boxes[pos_mask]
    pos_deltas = box_deltas[pos_mask]  # (P, num_classes * 4)

    # Select each positive box's own class-specific 4 deltas.
    pos_deltas = pos_deltas.view(-1, num_classes, 4) #reshapes into 3D grid: [number of boxes, number of classes, 4 coordinates] 
    idx = torch.arange(pos_deltas.size(0), device=pos_deltas.device) #list of integers to iterate through
    selected_deltas = pos_deltas[idx, pos_labels]  # (P, 4) - Extracts the coordinates of the box for the ground truth label
    #idx is the different sources, pos_labels tells you to only take the deltas of the boxes which correspond to the ground truth label

    # Finds the deltas the proposed boxes would need to match the ground truth boxes
    target_deltas = encode_boxes(pos_proposals, pos_gt_boxes)

    # Compares the predicted deltas to these actual target deltas
    # smooth L1 loss is loss function, as opposed to e.g. MSE
    reg_loss = F.smooth_l1_loss(selected_deltas, target_deltas)

    return cls_loss, reg_loss