# encoding: utf-8
from torch import nn

from layers import *
from utils import weights_init_classifier


class EmbeddingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        # classification layer
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'cosSoftmax':
            self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(features)
        else:
            cls_outputs = self.classifier(features, targets)

        return {
            "cls_outputs": cls_outputs,
            "features": features
        }
