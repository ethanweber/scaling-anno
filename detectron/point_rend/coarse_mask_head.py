# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm, interpolate
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.utils.events import get_event_storage


def mask_rcnn_min_loss(diverse_coarse_logits, instances, vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """

    min_loss = None
    min_index = None
    for index in range(len(diverse_coarse_logits)):
        mask_coarse_logits = diverse_coarse_logits[index]
        # TODO(ethan): bring back the vis_period to see what's going on during training
        loss = mask_rcnn_loss(mask_coarse_logits, instances, vis_period=0)

        if min_loss is None or loss < min_loss:
            min_loss = loss
            min_index = index

    get_event_storage().put_scalar("point_rend/coarse_min_index", min_index)
    return loss, min_index


def mask_iou_loss(iou_features, diverse_coarse_logits, instances, model, vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """

    iou_gt = []
    iou_predictions = []

    num_masks = diverse_coarse_logits.shape[1]
    for index in range(num_masks):
        mask_features = iou_features
        pred_mask_logits = diverse_coarse_logits[:, index]

        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = 14  # pred_mask_logits.size(2) TODO(ethan): fix this
        assert pred_mask_logits.size(2) == pred_mask_logits.size(
            3), "Mask prediction must be square!"

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(
                    dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
            mask_features = iou_features[:]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]
            mask_features = iou_features[indices]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Interactive_Image_Segmentation_CVPR_2018_paper.pdf

        pred_mask = interpolate(pred_mask_logits.view(-1, 1, 7, 7),
                                size=(14, 14),
                                mode="bilinear",
                                align_corners=False
                                )
        pred_masks = pred_mask.sigmoid()
        N = pred_masks.size(0)
        A = pred_masks.view(N, -1)
        B = gt_masks.view(N, -1)
        maximum = torch.sum(torch.max(A, B), dim=1)
        minimum = torch.sum(torch.min(A, B), dim=1)
        iou = minimum / (maximum + 1e-6)
        iou_gt.append(iou)

        predicted_iou, _ = model(
            mask_features.view(-1, 256, 14, 14),
            pred_masks.view(-1, 1, 14, 14)
        )
        predicted_iou = predicted_iou.view(-1)
        iou_predictions.append(predicted_iou)

    iou_predictions = torch.stack(iou_predictions, dim=1)
    iou_gt = torch.stack(iou_gt, dim=1)
    return iou_predictions, iou_gt


def mask_iou_loss_inference(iou_features, diverse_coarse_logits, instances, model, vis_period=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """

    iou_predictions = []
    iou_embeddings = []

    num_masks = diverse_coarse_logits.shape[1]
    for index in range(num_masks):
        mask_features = iou_features
        pred_mask_logits = diverse_coarse_logits[:, index]

        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = 14  # pred_mask_logits.size(2) TODO(ethan): fix this
        assert pred_mask_logits.size(2) == pred_mask_logits.size(
            3), "Mask prediction must be square!"

        pred_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            pred_classes_per_image = instances_per_image.pred_classes.to(
                dtype=torch.int64)
            pred_classes.append(pred_classes_per_image)

        if len(pred_classes) == 0:
            return None, None

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
            mask_features = iou_features[:]
        else:
            indices = torch.arange(total_num_masks)
            pred_classes = cat(pred_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, pred_classes]
            mask_features = iou_features[indices]

        # -- run the iou model ---
        pred_mask = interpolate(pred_mask_logits.view(-1, 1, 7, 7),
                                size=(14, 14),
                                mode="bilinear",
                                align_corners=False
                                )
        pred_masks = pred_mask.sigmoid()
        predicted_iou, embedding = model(
            mask_features.view(-1, 256, 14, 14),
            pred_masks.view(-1, 1, 14, 14)
        )
        predicted_iou = predicted_iou.view(-1)
        embedding = embedding.view(len(predicted_iou), -1)
        iou_predictions.append(predicted_iou)
        iou_embeddings.append(embedding)

    iou_predictions = torch.stack(iou_predictions, dim=1)
    iou_embeddings = torch.stack(iou_embeddings, dim=1)
    return iou_predictions, iou_embeddings


@ROI_MASK_HEAD_REGISTRY.register()
class CoarseMaskHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dim: the output dimension of the conv layers
            fc_dim: the feature dimenstion of the FC layers
            num_fc: the number of FC layers
            output_side_resolution: side resolution of the output square mask prediction
        """
        super(CoarseMaskHead, self).__init__()

        # fmt: off
        self.num_classes            = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dim                    = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.fc_dim                 = cfg.MODEL.ROI_MASK_HEAD.FC_DIM
        num_fc                      = cfg.MODEL.ROI_MASK_HEAD.NUM_FC
        self.output_side_resolution = cfg.MODEL.ROI_MASK_HEAD.OUTPUT_SIDE_RESOLUTION
        self.input_channels         = input_shape.channels
        self.input_h                = input_shape.height
        self.input_w                = input_shape.width
        # fmt: on

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                activation=F.relu,
            )
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = Conv2d(
            conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True, activation=F.relu
        )
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * \
            self.output_side_resolution * self.output_side_resolution

        self.prediction = nn.Linear(self.fc_dim, output_dim)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        # unlike BaseMaskRCNNHead, this head only outputs intermediate
        # features, because the features will be used later by PointHead.
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N, self.num_classes, self.output_side_resolution, self.output_side_resolution
        )


class MaskIouBranch(nn.Module):
    """
    """

    def __init__(self):
        super().__init__()
        self.maskiou_fcn1 = nn.Conv2d(257, 256, 3, 1, 1)
        self.maskiou_fcn2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maskiou_fcn4 = nn.Conv2d(256, 256, 3, 2, 1)
        self.maskiou_fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)
        self.maskiou_fc3 = nn.Linear(1024, 1)

        for l in [self.maskiou_fcn1, self.maskiou_fcn2, self.maskiou_fcn3, self.maskiou_fcn4]:
            nn.init.kaiming_normal_(
                l.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(l.bias, 0)
        for l in [self.maskiou_fc1, self.maskiou_fc2, self.maskiou_fc3]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, mask):
        """
        x -    [N, 256, 14, 14]
        mask - [N,   1, 14, 14]
        """
        # x are the same features as in CoarseMaskHead forward
        N = x.shape[0]
        x = torch.cat((x, mask), 1)
        x = F.relu(self.maskiou_fcn1(x))
        x = F.relu(self.maskiou_fcn2(x))
        x = F.relu(self.maskiou_fcn3(x))
        x = F.relu(self.maskiou_fcn4(x))
        x = x.view(N, -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        embedding = x.detach().clone()
        x = self.maskiou_fc3(x)
        x = torch.sigmoid(x)
        return x, embedding
