# mmdetection/configs/fire/faster-rcnn_swin-t_fpn_fire.py
default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        'mmdet.models.backbones.bricks.moe_mlp',
        'mmdet.models.backbones.swin_moe',
        'mmdet.models.detectors.faster_rcnn_moe',   # NEW

    ],
    allow_failed_imports=False)

_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',   
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
]

model = dict(
    type='FasterRCNNMoE',
    backbone=dict(
        _delete_=True,
        type='SwinTransformerMoE',        # uses shared-expand + gated expert-contract FFN
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,

        # MoE knobs (tune later if needed)
        num_experts=4,
        topk=2,                   # try 1 for max stability; 2 can help mixed domains
        moe_drop=0.0,
        gate_temperature=1.0,     # lower -> sharper routing; consider annealing during warmup
        use_prob_scale=True,
        balance_loss_weight=0.01,   
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'
        ),
    ),

    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        )
    )
)

classes = ('fire',)
data_root = 'data/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='bbox',
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=12)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1),
]

optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        }
    )
)

auto_scale_lr = dict(enable=False, base_batch_size=16)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    logger=dict(type='LoggerHook', interval=50),
)

work_dir = 'work_dirs/faster-rcnn_swin-t_fpn_fire'
