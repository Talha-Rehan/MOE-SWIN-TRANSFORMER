# Fire Detection with MMDetection

## Setup and Run

1. **Navigate to the mmdetection directory:**
```bash
cd mmdetection
conda activate mmdet
python tools/train.py configs/fire/faster-rcnn_swin-t_fpn_fire.py \
  --cfg-options \
    train_cfg.max_epochs=24 \
    param_scheduler.1.milestones="[16,22]" \
    optim_wrapper.optimizer.lr=5e-05


![alt text](image.png)