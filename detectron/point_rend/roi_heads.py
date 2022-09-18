# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.layers import ShapeSpec, cat, interpolate
from detectron2.modeling import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.modeling.roi_heads.mask_head import (
    build_mask_head,
    mask_rcnn_inference,
)
from detectron2.modeling.roi_heads.roi_heads import (
    select_foreground_proposals,
    StandardROIHeads
)
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head

from .point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
)
from .point_head import (
    build_point_head,
    roi_mask_point_loss,
    roi_mask_point_min_loss,
    roi_mask_point_diversity_loss
)

from .coarse_mask_head import (
    mask_iou_loss,
    mask_iou_loss_inference,
    mask_rcnn_min_loss,
    MaskIouBranch
)


from detectron2.utils.events import get_event_storage


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        print(prob.shape)
        instances.pred_masks = prob  # (1, Hmask, Wmask)


def mask_rcnn_loss(pred_mask_logits, instances, vis_period=0):
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
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
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
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() /
                         max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum(
    ).item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks, reduction="none")
    # mask_loss is [N, 7, 7]
    N = mask_loss.shape[0]
    mask_loss = torch.mean(mask_loss.view(N, -1), dim=1)
    # now [N]
    return mask_loss


def calculate_uncertainty(logits, classes):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, C, ...) or (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
        classes (list): A list of length R that contains either predicted of ground truth class
            for eash predicted mask.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


def calculate_uncertainty_inference(logits):
    """Fast version with predicted ones only"""
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


@ROI_HEADS_REGISTRY.register()
class PointRendStandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.projection_attention = torch.from_numpy(np.load(
            "/data/vision/torralba/scratch/ethanweber/scaleade/detectron/projection_matrices/features_attention_512.npy")).float()
        self.projection_backbone = torch.from_numpy(np.load(
            "/data/vision/torralba/scratch/ethanweber/scaleade/detectron/projection_matrices/features_backbone_512.npy")).float()
        self.projection_mask = torch.from_numpy(np.load(
            "/data/vision/torralba/scratch/ethanweber/scaleade/detectron/projection_matrices/features_mask_512.npy")).float()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["mask_head"] = build_mask_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["keypoint_head"] = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels,
                           width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        return_extras=False
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = \
                self.forward_with_given_boxes(
                    features,
                    pred_instances,
                    return_extras=return_extras)
            return pred_instances, {}

    def forward_with_given_boxes(
        self,
        features: Dict[str, torch.Tensor],
        instances: List[Instances],
        return_extras=False
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has(
            "pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(
            features,
            instances,
            return_extras=return_extras
        )
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals)
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class PointRendROIHeads(PointRendStandardROIHeads):
    """
    The RoI heads class for PointRend instance segmentation models.

    In this class we redefine the mask head of `StandardROIHeads` leaving all other heads intact.
    To avoid namespace conflict with other heads we use names starting from `mask_` for all
    variables that correspond to the mask head in the class's namespace.
    """

    def __init__(self, cfg, input_shape):
        # TODO use explicit args style
        super().__init__(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)

        self.mask_iou_branch = MaskIouBranch()
        self.num_masks = cfg.MODEL.POINT_HEAD.NUM_HEADS

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        # fmt: on

        self._init_coarse_head(cfg, input_shape)
        self._init_point_head(cfg, input_shape)

    def _init_coarse_head(self, cfg, input_shape):

        self.mask_coarse_in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        self.mask_coarse_side_size = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self._feature_scales = {
            k: 1.0 / v.stride for k, v in input_shape.items()}

        in_channels = np.sum(
            [input_shape[f].channels for f in self.mask_coarse_in_features])

        # NOTE(ethan): modification
        mask_coarse_heads = []
        # TODO(ethan): change the naming away from POINT_HEAD.NUM_HEADS since coarse
        for i in range(cfg.MODEL.POINT_HEAD.NUM_HEADS):
            mask_coarse_heads.append(
                build_mask_head(
                    cfg,
                    ShapeSpec(
                        channels=in_channels,
                        width=self.mask_coarse_side_size,
                        height=self.mask_coarse_side_size,
                    ),
                )
            )
        self.mask_coarse_heads = nn.ModuleList(mask_coarse_heads)
        # self.mask_coarse_head = build_mask_head(
        #     cfg,
        #     ShapeSpec(
        #         channels=in_channels,
        #         width=self.mask_coarse_side_size,
        #         height=self.mask_coarse_side_size,
        #     ),
        # )

        # print(self.mask_coarse_head)
        # import sys
        # sys.exit()

    def _init_point_head(self, cfg, input_shape):
        # fmt: off
        self.mask_point_on = cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON
        if not self.mask_point_on:
            return
        assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        self.mask_point_in_features = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.mask_point_train_num_points = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.mask_point_oversample_ratio = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.mask_point_importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        # next two parameters are use in the adaptive subdivions inference procedure
        self.mask_point_subdivision_steps = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.mask_point_subdivision_num_points = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum(
            [input_shape[f].channels for f in self.mask_point_in_features])

        self.mask_point_head = build_point_head(
            cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def _forward_mask(
            self,
            features,
            instances,
            return_extras=False):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            proposals, _ = select_foreground_proposals(
                instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features, diverse_coarse_logits = self._forward_mask_coarse(
                features,
                proposal_boxes
            )

            # add the mask loss
            coarse_losses = []
            for i in range(self.num_masks):
                temp = mask_rcnn_loss(
                    diverse_coarse_logits[:, i],
                    proposals
                )
                coarse_losses.append(temp)
            coarse_losses = torch.stack(coarse_losses, dim=1)
            indices = torch.argmin(coarse_losses, dim=1).long()
            # print(indices)

            # hindsight loss
            loss_mask = coarse_losses[
                torch.arange(coarse_losses.size(0)),
                indices
            ].mean()

            loss_point = self._forward_mask_point(
                features,
                diverse_coarse_logits[
                    torch.arange(diverse_coarse_logits.size(0)),
                    indices
                ],
                proposals,
            )

            # iou loss
            batch_indices = torch.arange(diverse_coarse_logits.size(0))
            mask_indices = torch.randint(0, self.num_masks, (diverse_coarse_logits.size(0),))
            iou_diverse_logits = diverse_coarse_logits.detach()[batch_indices, mask_indices, :, :, :].unsqueeze(1)
            iou_predictions, iou_gt = mask_iou_loss(
                mask_features.detach(),
                iou_diverse_logits,
                proposals,
                self.mask_iou_branch
            )
            iou_diff = torch.norm(iou_predictions - iou_gt, dim=1)
            loss_iou = iou_diff.mean()

            losses = {
                "loss_mask": loss_mask,
                "loss_point": loss_point,
                "loss_iou": loss_iou
            }
            return losses

        else:

            pred_boxes = [x.pred_boxes for x in instances]
            class_pred = cat([i.pred_classes for i in instances])
            mask_features, diverse_coarse_logits = self._forward_mask_coarse(
                features,
                pred_boxes
            )
            batch_indices = torch.arange(diverse_coarse_logits.size(0))

            # select best indices based on the best iou
            iou_predictions, iou_embeddings = mask_iou_loss_inference(
                mask_features,
                diverse_coarse_logits,
                instances,
                self.mask_iou_branch
            )

            if iou_predictions is not None:  # if None, means no instances detected
                indices = torch.argmax(iou_predictions, dim=1)

                mask_logits = []
                for mask_index in range(self.num_masks):
                    m = self._forward_mask_point(
                        features,
                        diverse_coarse_logits[
                            :,
                            mask_index
                        ],
                        instances,
                    )
                    mask_logits.append(m)
                mask_logits = torch.stack(mask_logits, dim=1)

                # add to Instances object
                num_boxes_per_image = [len(i) for i in instances]
                iou_predictions = iou_predictions.split(num_boxes_per_image, dim=0)
                iou_embeddings = iou_embeddings.split(num_boxes_per_image, dim=0)
                mask_logits = mask_logits.split(num_boxes_per_image, dim=0)
                mask_features = interpolate(
                    mask_features.view(-1, 256, 14, 14),
                    size=(7, 7),
                    mode="bilinear",
                    align_corners=False)
                mask_features = mask_features.split(num_boxes_per_image, dim=0)
                for inst, pred_ious, iou_embedding, pred_mask, mask_feat in zip(instances, iou_predictions, iou_embeddings, mask_logits, mask_features):
                    inst.pred_ious = pred_ious
                    inst.pred_masks = pred_mask.sigmoid()
                    features_logits_fine = pred_mask.sigmoid()

                    # TODO(ethan): check if this still works for multiple heads?
                    small_features_logits_fine = interpolate(
                        features_logits_fine.view(-1, 1, 56, 56),
                        size=(7, 7),
                        mode="bilinear",
                        align_corners=False)
                    if small_features_logits_fine.shape[0] != 0:
                        # print(small_features_logits_fine.shape)
                        # print(mask_feat.shape)
                        features_attention = mask_feat * small_features_logits_fine
                        self.projection_attention = self.projection_attention.to(features_attention.device)
                        self.projection_backbone = self.projection_backbone.to(features_attention.device)
                        self.projection_mask = self.projection_mask.to(features_attention.device)

                        inst.features_attention = torch.matmul(features_attention.view(
                            features_attention.shape[0], -1), self.projection_attention)
                        inst.features_mask = torch.matmul(features_logits_fine.view(
                            features_logits_fine.shape[0], -1), self.projection_mask)
                        inst.features_backbone = torch.matmul(mask_feat.view(
                            mask_feat.shape[0], -1), self.projection_backbone)
                        inst.features_iou_embedding = iou_embedding.view(iou_embedding.shape[0], -1)

            if return_extras:
                raise ValueError("return_extras is not longer supported!")

            return instances

    def _forward_mask_coarse(self, features, boxes):
        """
        Forward logic of the coarse mask head.
        """
        point_coords = generate_regular_grid_point_coords(
            np.sum(len(x)
                   for x in boxes), self.mask_coarse_side_size, boxes[0].device
        )
        mask_coarse_features_list = [features[k]
                                     for k in self.mask_coarse_in_features]
        features_scales = [self._feature_scales[k]
                           for k in self.mask_coarse_in_features]
        # For regular grids of points, this function is equivalent to `len(features_list)' calls
        # of `ROIAlign` (with `SAMPLING_RATIO=2`), and concat the results.
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, features_scales, boxes, point_coords
        )

        # diverse_coarse_logits = self.mask_coarse_head(mask_features)
        diverse_coarse_logits = []
        for mask_coarse_head in self.mask_coarse_heads:
            coarse_logits = mask_coarse_head(mask_features)
            diverse_coarse_logits.append(coarse_logits)
        diverse_coarse_logits = torch.stack(diverse_coarse_logits, dim=1)

        return mask_features, diverse_coarse_logits

    def _forward_mask_point(self, features, mask_coarse_logits, instances):
        """
        Forward logic of the mask point head.
        """
        if not self.mask_point_on:
            return {} if self.training else mask_coarse_logits

        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._feature_scales[k]
                           for k in self.mask_point_in_features]

        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            gt_classes = cat([x.gt_classes for x in instances])
            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    mask_coarse_logits,
                    lambda logits: calculate_uncertainty(logits, gt_classes),
                    self.mask_point_train_num_points,
                    self.mask_point_oversample_ratio,
                    self.mask_point_importance_sample_ratio,
                )

            fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
                mask_features_list, features_scales, proposal_boxes, point_coords
            )
            coarse_features = point_sample(
                mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(
                fine_grained_features, coarse_features)
            return roi_mask_point_loss(
                point_logits, instances, point_coords_wrt_image
            )
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            pred_classes = cat([x.pred_classes for x in instances])
            # The subdivision code will fail with the empty list of boxes
            if len(pred_classes) == 0:
                return mask_coarse_logits

            # NOTE(ethan): faster than the original implementation
            batch_indices = torch.arange(mask_coarse_logits.size(0))
            mask_logits = mask_coarse_logits.clone()[batch_indices, pred_classes].unsqueeze(1)
            for subdivions_step in range(self.mask_point_subdivision_steps):
                mask_logits = interpolate(
                    mask_logits, scale_factor=2, mode="bilinear", align_corners=False
                )
                # If `mask_point_subdivision_num_points` is larger or equal to the
                # resolution of the next step, then we can skip this step
                H, W = mask_logits.shape[-2:]
                if (
                    self.mask_point_subdivision_num_points >= 4 * H * W
                    and subdivions_step < self.mask_point_subdivision_steps - 1
                ):
                    continue
                uncertainty_map = calculate_uncertainty(mask_logits, None)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.mask_point_subdivision_num_points
                )
                fine_grained_features, _ = point_sample_fine_grained_features(
                    mask_features_list, features_scales, pred_boxes, point_coords
                )
                coarse_features = point_sample(
                    mask_coarse_logits, point_coords, align_corners=False
                )

                # NOTE(ethan): I had to fix this too
                point_logits = self.mask_point_head(
                    fine_grained_features, coarse_features)[batch_indices, pred_classes].unsqueeze(1)

                # put mask point predictions to the right places on the upsampled grid.
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(R, C, H, W)
                )
            assert mask_logits.shape[1] == 1
            return mask_logits[:, 0, :, :]
