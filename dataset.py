import torch
from torch.utils.data import Dataset
from cutouts import SamplesPreprocessor
import numpy as np


class RadioGalaxyDataset(Dataset):
    """
    Wraps existing PyBDSF-based proposal generation and official-catalogue
    ground-truth generation into a standard PyTorch Dataset.

    Each __getitem__ call must return, for ONE image:
        image:      FloatTensor, shape (C, H, W)
        proposals:  FloatTensor, shape (N, 4)  -- PyBDSF candidate boxes, [x1,y1,x2,y2]
        gt_boxes:   FloatTensor, shape (M, 4)  -- official catalogue ground-truth boxes
        gt_labels:  LongTensor,  shape (M,)    -- class id per ground-truth box (1..num_classes-1)
    """

    def __init__(self, catalogue_path, cutout_dir, transform=None):
        """
        Args:
            catalogue_path: path to the official catalogue used for ground truth.
            cutout_dir: directory containing (or used to generate) FITS cutouts.
            transform: optional callable applied to the image tensor (e.g. normalization).
        """
        self.catalogue_path = catalogue_path
        self.cutout_dir = cutout_dir
        self.transform = transform

        # TODO: wire up dynamic CLI paths for multi-facet scaling

        #samples = [{fits_file, pybdsf_proposals, ground truth boxes, ground truth labels},{...},...]        
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_processor = SamplesPreprocessor()
        return sample_processor.generate_samples_list()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Each __getitem__ call must return, for ONE image:
            image:      FloatTensor, shape (C, H, W)
            proposals:  FloatTensor, shape (N, 4)  -- PyBDSF candidate boxes, [x1,y1,x2,y2]
            gt_boxes:   FloatTensor, shape (M, 4)  -- official catalogue ground-truth boxes
            gt_labels:  LongTensor,  shape (M,)    -- class id per ground-truth box (1..num_classes-1)
        """
        
        sample = self.samples[idx]

        # 1. Load/generate the image cutout as a tensor, shape (C, H, W)
        image = self._load_image(sample)

        # 2. Generate PyBDSF candidate proposal boxes for this cutout
        proposals = self._generate_pybdsf_proposals(sample)

        # 3. Generate ground-truth boxes + labels from the official catalogue
        gt_boxes, gt_labels = self._generate_ground_truth(sample)

        if self.transform is not None:
            image = self.transform(image)

        return image, proposals, gt_boxes, gt_labels

    def _load_image(self, sample):
        image_array = np.load(sample["cutout_path"]) # Already preprocessed!
        image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0) #unsqueeze changes shape: (H,W) -> (1,H,W)
        return image
    
    def _generate_pybdsf_proposals(self, sample):
        """
        Extracts pre-computed PyBDSF candidate proposal boxes for this cutout.
        Returns:
            proposals: FloatTensor, shape (N, 4) -- [x1, y1, x2, y2]
        """
        proposals_np = sample["proposals"]
        return torch.tensor(proposals_np, dtype=torch.float32)

    def _generate_ground_truth(self, sample):
        """
        Extracts official catalogue ground-truth boxes and labels for this cutout.
        Returns:
            gt_boxes:  FloatTensor, shape (M, 4)
            gt_labels: LongTensor,  shape (M,)
        """
        gt_boxes_np = sample["gt_box"]
        gt_labels_np = sample["gt_label"]
        
        # Ensure correct tensor types and shapes (even if empty)
        gt_boxes = torch.tensor(gt_boxes_np, dtype=torch.float32)
        
        # Handle label shaping depending on whether it's a single value or an array
        if isinstance(gt_labels_np, (int, np.integer)):
            gt_labels = torch.tensor([gt_labels_np], dtype=torch.long)
        else:
            gt_labels = torch.tensor(gt_labels_np, dtype=torch.long)
            
        return gt_boxes, gt_labels


def collate_fn(batch):
    """
    Custom collate function required because each image can have a different
    number of proposal boxes and ground-truth boxes.

    Images are stacked normally (they're all the same fixed size after cutout
    generation). Boxes and labels stay as per-image lists, matching the format
    TinyFastRCNN.forward() and box_iou-based matching both expect.
    """
    images, proposals, gt_boxes, gt_labels = zip(*batch)
    images = torch.stack(images, dim=0)
    proposals = list(proposals)
    gt_boxes = list(gt_boxes)
    gt_labels = list(gt_labels)
    return images, proposals, gt_boxes, gt_labels