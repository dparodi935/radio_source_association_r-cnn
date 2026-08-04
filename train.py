import torch
from torch.utils.data import DataLoader

from model import TinyFastRCNN
from dataset import RadioGalaxyDataset, collate_fn
from losses import match_boxes_to_gt, compute_losses
import os

def train(
    catalogue_path,
    cutout_dir,
    num_classes=3,
    in_channels=1,
    batch_size=4,
    num_epochs=20,
    lr=1e-4,
    iou_threshold=0.3,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RadioGalaxyDataset(catalogue_path, cutout_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_cls_loss = 0.0
        running_reg_loss = 0.0

        for images, proposals, gt_boxes, gt_labels in dataloader:
            images = images.to(device)
            proposals = [p.to(device) for p in proposals]
            gt_boxes = [g.to(device) for g in gt_boxes]
            gt_labels = [l.to(device) for l in gt_labels]

            # Forward pass: model handles the whole batch of images/proposals at once
            cls_logits, box_deltas = model(images, proposals)
            
            if torch.isnan(cls_logits).any():
                print("WARNING: cls_logits contains NaN! The backbone or RoIAlign produced bad features.")
                break
            if torch.isnan(box_deltas).any():
                print("WARNING: box_deltas contains NaN!")
                break

            # Matching happens per image, since IoU only makes sense within one
            # image's own proposals vs that image's own ground truth. Results
            # are concatenated to line up with cls_logits/box_deltas, which are
            # already flattened across the batch by RoIAlign.
            all_matched_labels = []
            all_matched_proposals = []
            all_matched_gt_boxes = []

            for img_proposals, img_gt_boxes, img_gt_labels in zip(proposals, gt_boxes, gt_labels):
                matched_labels, matched_gt_boxes = match_boxes_to_gt(
                    img_proposals, img_gt_boxes, img_gt_labels, iou_threshold=iou_threshold
                )
                all_matched_labels.append(matched_labels)
                all_matched_proposals.append(img_proposals)
                all_matched_gt_boxes.append(matched_gt_boxes)

            matched_labels = torch.cat(all_matched_labels, dim=0)
            matched_proposals = torch.cat(all_matched_proposals, dim=0)
            matched_gt_boxes = torch.cat(all_matched_gt_boxes, dim=0)

            cls_loss, reg_loss = compute_losses(
                cls_logits,
                box_deltas,
                matched_labels,
                matched_proposals,
                matched_gt_boxes,
                num_classes=num_classes,
            )
            loss = cls_loss + reg_loss


            optimizer.zero_grad() #clears .grad
            
            loss.backward() #pupoulates .grad
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #rescales .grad
            
            optimizer.step() #consumes .grad to update weights


            running_cls_loss += cls_loss.item()
            running_reg_loss += reg_loss.item()

        n_batches = len(dataloader)
        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"- cls_loss: {running_cls_loss / n_batches:.4f} "
            f"- reg_loss: {running_reg_loss / n_batches:.4f}"
        )

    return model


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trained_model = train(
        catalogue_path="path/to/official_catalogue.csv",
        cutout_dir="path/to/cutouts",
        num_classes=3,
        in_channels=1,
        batch_size=4,
        num_epochs=20,
        lr=1e-4,
    )
    print("Model trained")
    torch.save(trained_model.state_dict(), os.path.join(script_dir, "tiny_fastrcnn_weights.pt"))
    print("Weights saved")