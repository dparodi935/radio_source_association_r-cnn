import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def listwise_loss(cls_logits, proposals, gt_boxes, iou_thresh=0.9):
    """One softmax per cutout over ITS OWN proposals.

    cls_logits: (total_proposals, num_classes   ) concatenated across the batch, in the same
                order RoIAlign produced them (image 0's boxes, then image 1's...). 
                Consists of the scores for each class for each proposal
    proposals:  list of (n_i, 4) tensors, one per image
    gt_boxes:   list of (0|1, 4) tensors, one per image
    """
    losses = []
    start = 0
    
    for props, gt in zip(proposals, gt_boxes):
        end = start + len(props)
        if len(gt) > 0:
            ious = box_iou(props, gt)[:, 0]
            target = int(torch.argmax(ious)) # chooses best proposal
            if float(ious[target]) >= iou_thresh: # check if best proposal's iou is above the threshold, otherwise it won't train on it
                # .max below is a no-op: for case of num_classes=2 returns maximum of one element list (since ,1: removes bg class)
                # only relevant if num_classes > 2
                score = cls_logits[start:end, 1:].max(dim=1).values  # (n_i,) . Returns list of raw scores from  the relevant proposals
                
                # normalises with softmax over list of proposals
                # score[None, :] reshapes from (n_props, ) to (1, n_props) ie. one sample with n_props classes. cross_entropy normalises over 'classes'
                # this is instead of a sigmoid per proposal, which scores the proposals independently
                losses.append(F.cross_entropy(score[None, :], torch.tensor([target], device=score.device)))
        start = end
        
    if not losses:
        return cls_logits.sum() * 0.0
    
    return torch.stack(losses).mean()