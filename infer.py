import argparse
import os

import torch
import torch.nn.functional as F
from torchvision.ops import nms

from model import TinyFastRCNN
from losses import decode_boxes
from dataset import RadioGalaxyDataset


def predict(model, image, proposals, score_threshold=0.5, nms_iou_threshold=0.3, device=None):
    """
    Runs inference on a single image with its candidate proposal boxes.

    Args:
        model: trained TinyFastRCNN
        image: FloatTensor, shape (1, C, H, W)  -- single image, batch dim of 1
        proposals: FloatTensor, shape (N, 4)    -- PyBDSF candidate boxes for this image
        score_threshold: minimum class confidence to keep a detection
        nms_iou_threshold: IoU threshold for suppressing duplicate detections

    Returns:
        final_boxes:  (K, 4) kept boxes after thresholding + NMS
        final_scores: (K,)   confidence score of the kept class per box
        final_labels: (K,)   predicted class id per box
    """
    device = device or next(model.parameters()).device
    model.eval()

    image = image.to(device)
    proposals = proposals.to(device)

    with torch.no_grad(): #turn off gradient tracking, the mechanism used to update the weights during training
        # feed the cutout and the proposals to the model. Output: raw class scores, raw box adjustments
        # WHERE ACTUAL INFERENCE HAPPENS
        cls_logits, box_deltas = model(image, [proposals]) 

        #scales the class scores into clean percentages
        scores = F.softmax(cls_logits, dim=1)          # (N, num_classes). dim=1 so it reads the entire row (instead of dim=0 for a column)
        num_classes = scores.size(1) # Extracts the number of classes (in this case 3) from tensor 

        # Best non-background class per proposal
        fg_scores, fg_labels = scores[:, 1:].max(dim=1) # finds the highest non-background (use [:, 1:] to isolate) score 
        fg_labels = fg_labels + 1  # shift back since we sliced off background (class 0)

        # Select the box-delta slice matching each proposal's predicted class
        box_deltas = box_deltas.view(-1, num_classes, 4) #reshapes 1D array of box coordinates into 3D tensor
        idx = torch.arange(box_deltas.size(0), device=device) #creates integer list from 0 up to number of boxes
        selected_deltas = box_deltas[idx, fg_labels] #extracts box adjustments for the chosen class (ie. highest score non-bg)

        #Applies the box adjustments to the proposal
        decoded_boxes = decode_boxes(proposals, selected_deltas)

        #Confidence filtering - Calculates and applies mask
        keep_mask = fg_scores > score_threshold  # boolean mask that selects for scores above threshold
        boxes = decoded_boxes[keep_mask]
        scores = fg_scores[keep_mask]
        labels = fg_labels[keep_mask]

        # NMS per class, since boxes of different classes shouldn't suppress each other
        final_boxes, final_scores, final_labels = [], [], []
        for class_id in labels.unique():
            class_mask = (labels == class_id)  # creates mask that selects for a given class (single or MC)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            # nms: If boxes are highly overlapping, every box except the one with the highest score is removed
            # nms_iou_threshold determine what highly overlapping means
            keep = nms(class_boxes, class_scores, nms_iou_threshold)  
            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(torch.full((len(keep),), class_id, dtype=torch.long))

        if final_boxes:
            #if any boxes survived filtering, the python lists of tensors are merged into flat unified PyTorch tensor
            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            final_labels = torch.cat(final_labels, dim=0)
        else:
            final_boxes = torch.empty((0, 4))
            final_scores = torch.empty((0,))
            final_labels = torch.empty((0,), dtype=torch.long)

    return final_boxes, final_scores, final_labels


CLASS_NAMES = {0: "background", 1: "single", 2: "multi-component"}
CLASS_COLORS = {1: "lime", 2: "orange"}


def visualize_detections(image, boxes, scores, labels, save_path=None, title=None):
    """
    Draws kept detection boxes over a single-channel cutout and saves/shows it.

    Args:
        image: (1, H, W) or (H, W) tensor/array -- the cutout, on CPU
        boxes, scores, labels: CPU tensors, the outputs of predict()
        save_path: if given, saves the figure here instead of showing it
        title: optional plot title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed -- skipping visualization (pip install matplotlib)")
        return

    #squeeze removes the useless extra info from the tensor, creating a flat 2D grid. Then numpy turns into a np array
    img = image.squeeze().numpy() if torch.is_tensor(image) else image

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="lower")

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.tolist()  # the two coordinates for the box 
        color = CLASS_COLORS.get(int(label), "red")   #Looks at CLASS_COLORS to find colour, defaults to red if unable to
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                  edgecolor=color, facecolor="none")  #creates the box
        ax.add_patch(rect)
        ax.text(x1, y1 - 2, f"{CLASS_NAMES.get(int(label), int(label))} {score:.2f}", 
                color=color, fontsize=8, va="bottom") #labels the box with the score and class

    ax.set_title(title or "")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def run_inference(
    weights_path,
    catalogue_path,
    cutout_dir,
    num_classes=3,
    in_channels=1,
    score_threshold=0.5,
    nms_iou_threshold=0.3,
    sample_indices=None,
    output_dir="inference_outputs",
    device=None,
):
    """
    Loads a trained model and runs it over real dataset samples (real cutouts +
    real PyBDSF proposals), printing kept detections and saving a visualization
    for each sample.

    Args:
        weights_path: path to a .pt file saved by train.py
        catalogue_path, cutout_dir: forwarded to RadioGalaxyDataset. NOTE: as of
            now RadioGalaxyDataset ignores these and always loads whatever facet
            is hardcoded in SamplesPreprocessor (currently "facet_5") -- see the
            TODO in dataset.py. Wire these through properly before using this on
            a different facet.
        sample_indices: iterable of dataset indices to run inference on, or None
            to run over the whole dataset
        output_dir: where per-sample visualization PNGs are saved
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #creates the CNN model and loads the generated weights
    model = TinyFastRCNN(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # puts model in evaluation mode, switching off trianing specific layers like Dropout and BatchNorm

    dataset = RadioGalaxyDataset(catalogue_path, cutout_dir)
    # which indices to actually use. Uses all unless specified otherwise
    indices = sample_indices if sample_indices is not None else range(len(dataset))  

    os.makedirs(output_dir, exist_ok=True)

    results = []
    for idx in indices:
        image, proposals, gt_boxes, gt_labels = dataset[idx]
        image_batch = image.unsqueeze(0)  # From numpy array to (1, C, H, W), the batch-of-1 predict() expects

        boxes, scores, labels = predict(
            model, image_batch, proposals,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=device,
        )
        boxes, scores, labels = boxes.cpu(), scores.cpu(), labels.cpu() # If tensors processed on GPU, this copies them to the CPU

        #prints info about each image
        print(f"Sample {idx}: {len(boxes)} detection(s)")
        for box, score, label in zip(boxes, scores, labels):
            name = CLASS_NAMES.get(int(label), int(label))
            print(f"  {name:>16s}  score={score:.3f}  box={[round(v, 1) for v in box.tolist()]}")

        #creates matplotlib image visualising the source and the detection box
        visualize_detections(
            image, boxes, scores, labels,
            save_path=os.path.join(output_dir, f"sample_{idx}.png"),
            title=f"Sample {idx} ({len(boxes)} detections)",
        )

        results.append({"idx": idx, "boxes": boxes, "scores": scores, "labels": labels})

    print(f"\nSaved {len(results)} visualization(s) to {output_dir}/")
    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Run TinyFastRCNN inference on real cutouts")
    parser.add_argument("--weights", default=os.path.join(script_dir,"tiny_fastrcnn_weights.pt"),
                         help="Path to trained weights saved by train.py")
    parser.add_argument("--catalogue-path", default="path/to/official_catalogue.csv")
    parser.add_argument("--cutout-dir", default="path/to/cutouts")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=50,
                         help="Number of dataset samples to run inference on (-1 for all)")
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "inference_outputs"))
    args = parser.parse_args()

    sample_indices = None if args.num_samples == -1 else range(args.num_samples)

    run_inference(
        weights_path=args.weights,
        catalogue_path=args.catalogue_path,
        cutout_dir=args.cutout_dir,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        sample_indices=sample_indices,
        output_dir=args.output_dir,
    )