# mmdet/models/detectors/faster_rcnn_moe.py
from __future__ import annotations
from mmengine.registry import MODELS
from mmdet.models.detectors import FasterRCNN

@MODELS.register_module()
class FasterRCNNMoE(FasterRCNN):
    """Faster R-CNN that also aggregates MoE aux losses from the backbone."""
    def forward_train(self, imgs, img_metas, **kwargs):
        # call the standard pipeline to produce detection losses
        losses = super().forward_train(imgs, img_metas, **kwargs)

        # collect all aux losses exposed by MoE modules in the backbone
        aux = None
        for m in self.backbone.modules():
            if hasattr(m, 'get_aux_loss'):
                v = m.get_aux_loss(reset=True)  # tensor scalar
                aux = v if aux is None else (aux + v)

        if aux is not None:
            # ensure it's in the same device and dtype
            losses['loss_moe_aux'] = aux

        return losses
