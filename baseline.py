# encoding: utf-8
import torch
from torch import nn

from backbones import Backbone, MobileFaceNet
from heads import EmbeddingHead
from torch.nn import CrossEntropyLoss
from losses import *


class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer(
            "pixel_std",
            torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        if cfg.MODEL.BACKBONE.NAME == 'resnet':
            # self.backbone = build_resnet_backbone(cfg)
            self.backbone = Backbone(cfg.MODEL.BACKBONE.DEPTH, cfg.MODEL.BACKBONE.DROP_RATIO, cfg.MODEL.BACKBONE.NET_MODE)
        elif cfg.MODEL.BACKBONE.NAME == 'mobilenet':
            self.backbone = MobileFaceNet(cfg)

        # head
        self.heads = EmbeddingHead(cfg)
        
        self.ce_loss = CrossEntropyLoss()

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, inputs, targets=None):
        images = self.preprocess_image(inputs)
        features = self.backbone(images)
        if self.training:
            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            return features

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError(
                "batched_inputs must be dict or torch.Tensor, but get {}".
                format(type(batched_inputs)))

        # images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            # loss_dict["loss_cls"] = cross_entropy_loss(
            #     cls_outputs,
            #     gt_labels,
            #     self._cfg.MODEL.LOSSES.CE.EPSILON,
            #     self._cfg.MODEL.LOSSES.CE.ALPHA,
            # ) * self._cfg.MODEL.LOSSES.CE.SCALE
            loss_dict["loss_cls"] = self.ce_loss(
                cls_outputs,
                gt_labels,
            )

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE
        
        if "Focal" in loss_names:
            loss_dict["loss_focal"] = focal_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.COSFACE.MARGIN,
                self._cfg.MODEL.LOSSES.COSFACE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.COSFACE.SCALE

        return loss_dict
