import torch
from torch.utils.data import Dataset
from cutouts import SamplesPreprocessor
import numpy as np

import random

from cutouts import SamplesPreprocessor, discover_mosaics


class RadioGalaxyDataset(Dataset):
    """Concatenates samples from an explicit list of mosaics."""

    def __init__(self, data_root, mosaic_ids, size=200, max_neighbours=8,
                 transform=None, verbose=True, rotations=(0,), encoding="radio3"):
        self.data_root = data_root
        self.mosaic_ids = list(mosaic_ids)
        self.transform = transform

        self.samples = []
        for mid in self.mosaic_ids:
            pre = SamplesPreprocessor(mid, data_root, size=size,
                                      max_neighbours=max_neighbours,
                                      rotations=rotations, encoding=encoding)
            self.samples.extend(pre.generate_samples_list(verbose=verbose))

        if verbose:
            n_gt = sum(1 for s in self.samples if len(s["gt_box"]) > 0)
            n_mc = sum(1 for s in self.samples
                       if len(s["gt_label"]) and s["gt_label"][0] == 2)
            print(f"[dataset] {len(self.samples)} samples from "
                  f"{len(self.mosaic_ids)} mosaics | {n_gt} with GT | {n_mc} multi-component")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = torch.from_numpy(np.load(s["cutout_path"])).float()
        if self.transform is not None:
            image = self.transform(image)

        proposals = torch.from_numpy(np.ascontiguousarray(s["proposals"])).float()
        gt_boxes = torch.from_numpy(np.ascontiguousarray(s["gt_box"])).float()
        gt_labels = torch.from_numpy(np.ascontiguousarray(s["gt_label"])).long()

        assert proposals.ndim == 2 and proposals.shape[1] == 4, proposals.shape
        assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4, gt_boxes.shape
        assert gt_labels.shape[0] == gt_boxes.shape[0]

        return image, proposals, gt_boxes, gt_labels


def split_mosaics(data_root, fracs=(0.7, 0.15, 0.15), seed=42,
                  max_train=None, max_val=None, max_test=None):
    """Partition mosaic ids by whole mosaic, so no cutout leaks across splits."""
    ids = discover_mosaics(data_root)
    if not ids:
        raise RuntimeError(f"no mosaics found under {data_root}")
    random.Random(seed).shuffle(ids)

    n = len(ids)
    n_tr = max(1, int(round(fracs[0] * n)))
    n_va = int(round(fracs[1] * n))
    train, val, test = ids[:n_tr], ids[n_tr:n_tr + n_va], ids[n_tr + n_va:]

    if max_train is not None:
        train = train[:max_train]
    if max_val is not None:
        val = val[:max_val]
    if max_test is not None:
        test = test[:max_test]
    return train, val, test


def collate_fn(batch):
    images, proposals, gt_boxes, gt_labels = zip(*batch)
    return (torch.stack(images, 0), list(proposals),
            list(gt_boxes), list(gt_labels))
    
    
    
    
    
    
    
    
    
    
    
 ###################################    ################################### ################################### ###################################
    
