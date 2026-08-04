import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class TinyFastRCNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=1):
        """
        An educational Fast R-CNN model for astronomical FITS images.
        
        Args:
            num_classes: Number of target classes (e.g., 0=Background, 1=FRI, 2=FRII).
            in_channels: 1 for standard FITS continuum maps, 3 if stacking radio+IR.
        """
        super(TinyFastRCNN, self).__init__()
        
        # 1. THE BACKBONE (Feature Extractor) - Creates the feature map
        # Downsamples the image spatial dimensions by a factor of 8 (2^3)
        self.backbone = nn.Sequential(
            # Block 1: 128x126 -> 64x64 - THESE NUMBERS AREN'T ACCURATE SINCE I'VE MODIFIED CUTOUT SIZE
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), #output_channels=16 ie. 16 filters, 16 feature maps
            nn.BatchNorm2d(16), #Rescales the activations to have a mean of zero and variance of one
            nn.ReLU(), #applies max(0,x) to activations
            nn.MaxPool2d(kernel_size=2, stride=2), #downsampling, slides window across feature map and selects max value
            #pooling has 2x2 window with stride=2, so no overlap
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: (Final feature map channel depth = 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 2. THE ROI ALIGN LAYER
        # Maps arbitrary bounding boxes onto the 32x32 feature map and 
        # pools them into fixed 7x7 spatial tensors.
        # spatial_scale = 1.0 / 8.0 (because our backbone reduced dimensions by 8x)
        self.roi_align = ops.RoIAlign(
            output_size=(7, 7), 
            spatial_scale=0.25, 
            sampling_ratio=2
        )
        
        # 3. THE FULLY CONNECTED SHARED HEAD
        # Flattens the 7x7x64 pooled RoI features into dense layers 
        # 64 feature maps of dimension 7x7
        self.fc_shared = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256), #takes all the inputs  (7*7*64=3136) and connects them to 256 output neurons with y = Wx+b
            nn.ReLU(),
            nn.Dropout(0.5), # randomly turns off half of all neurons during training, to prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 4. THE MULTI-TASK HEADS
        # Head A: Classification (predicts probability across num_classes)
        self.cls_head = nn.Linear(128, num_classes)
        
        # Head B: Bounding Box Regression (predicts 4 adjustment deltas per class: dx, dy, dw, dh)
        # We output 4 * num_classes so each class gets its own customized box tweaks
        #BELIEVE THIS IS UNUSED BY MOSTERT ET AL. 2022
        self.reg_head = nn.Linear(128, num_classes * 4)

    def forward(self, images, boxes):
        """
        Args:
            images: Tensor of shape (Batch_Size, Channels, Height, Width)
            boxes: List of Tensors. Each tensor in the list corresponds to one image
                   in the batch and has shape (Num_Boxes, 4) in [x1, y1, x2, y2] format.
        
        Returns:
            cls_logits: Shape (Total_Boxes_In_Batch, num_classes)
            box_deltas: Shape (Total_Boxes_In_Batch, num_classes * 4)
        """
        # Step 1: Extract full-image feature maps in a single forward pass
        feature_maps = self.backbone(images)
        
        # Step 2: Pool variable-sized boxes into uniform 7x7x64 feature cubes
        # RoIAlign automatically handles formatting the box list for the batch
        pooled_rois = self.roi_align(feature_maps, boxes)
        
        # Step 3: Flatten spatial dimensions (Total_Boxes, 64*7*7) and run dense network
        flattened_rois = pooled_rois.view(pooled_rois.size(0), -1)
        shared_features = self.fc_shared(flattened_rois)
        
        # Step 4: Branch into simultaneous classification and regression
        cls_logits = self.cls_head(shared_features)
        box_deltas = self.reg_head(shared_features)
        
        return cls_logits, box_deltas