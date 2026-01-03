checkpoint_config = dict(interval=1)
data = dict(
    samples_per_gpu=64,
    test=dict(
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                size=(
                    256,
                    -1,
                ),
                type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    train=dict(
        data_prefix='data/imagenet/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                size=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', flip_prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                magnitude_level=9,
                magnitude_std=0.5,
                num_policies=2,
                policies=[
                    dict(type='AutoContrast'),
                    dict(type='Equalize'),
                    dict(type='Invert'),
                    dict(
                        magnitude_key='angle',
                        magnitude_range=(
                            0,
                            30,
                        ),
                        type='Rotate'),
                    dict(
                        magnitude_key='bits',
                        magnitude_range=(
                            4,
                            0,
                        ),
                        type='Posterize'),
                    dict(
                        magnitude_key='thr',
                        magnitude_range=(
                            256,
                            0,
                        ),
                        type='Solarize'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            110,
                        ),
                        type='SolarizeAdd'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='ColorTransform'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Contrast'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Brightness'),
                    dict(
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.9,
                        ),
                        type='Sharpness'),
                    dict(
                        direction='horizontal',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        direction='vertical',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.3,
                        ),
                        type='Shear'),
                    dict(
                        direction='horizontal',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                    dict(
                        direction='vertical',
                        magnitude_key='magnitude',
                        magnitude_range=(
                            0,
                            0.45,
                        ),
                        type='Translate'),
                ],
                total_level=10,
                type='RandAugment'),
            dict(
                erase_prob=0.25,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'gt_label',
            ], type='ToTensor'),
            dict(keys=[
                'img',
                'gt_label',
            ], type='Collect'),
        ],
        type='ImageNet'),
    val=dict(
        ann_file='data/imagenet/meta/val.txt',
        data_prefix='data/imagenet/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                size=(
                    256,
                    -1,
                ),
                type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                std=[
                    58.395,
                    57.12,
                    57.375,
                ],
                to_rgb=True,
                type='Normalize'),
            dict(keys=[
                'img',
            ], type='ImageToTensor'),
            dict(keys=[
                'img',
            ], type='Collect'),
        ],
        type='ImageNet'),
    workers_per_gpu=8)
dataset_type = 'ImageNet'
dist_params = dict(backend='nccl')
evaluation = dict(interval=10, metric='accuracy')
img_norm_cfg = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
    ], interval=100)
log_level = 'INFO'
lr_config = dict(
    by_epoch=False,
    min_lr_ratio=0.01,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=20,
    warmup_ratio=0.001)
model = dict(
    backbone=dict(
        arch='tiny', drop_path_rate=0.2, img_size=224, type='SwinTransformer'),
    head=dict(
        cal_acc=False,
        in_channels=768,
        init_cfg=None,
        loss=dict(
            label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
        num_classes=1000,
        type='LinearClsHead'),
    init_cfg=[
        dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
        dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
    ],
    neck=dict(type='GlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.8, num_classes=1000, prob=0.5, type='BatchMixup'),
        dict(alpha=1.0, num_classes=1000, prob=0.5, type='BatchCutMix'),
    ]),
    type='ImageClassifier')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.001,
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
        norm_decay_mult=0.0),
    type='AdamW',
    weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
paramwise_cfg = dict(
    bias_decay_mult=0.0,
    custom_keys=dict({
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    }),
    norm_decay_mult=0.0)
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(magnitude_key='angle', magnitude_range=(
        0,
        30,
    ), type='Rotate'),
    dict(magnitude_key='bits', magnitude_range=(
        4,
        0,
    ), type='Posterize'),
    dict(magnitude_key='thr', magnitude_range=(
        256,
        0,
    ), type='Solarize'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            110,
        ),
        type='SolarizeAdd'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='ColorTransform'),
    dict(
        magnitude_key='magnitude', magnitude_range=(
            0,
            0.9,
        ), type='Contrast'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='Brightness'),
    dict(
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.9,
        ),
        type='Sharpness'),
    dict(
        direction='horizontal',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.3,
        ),
        type='Shear'),
    dict(
        direction='vertical',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.3,
        ),
        type='Shear'),
    dict(
        direction='horizontal',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.45,
        ),
        type='Translate'),
    dict(
        direction='vertical',
        magnitude_key='magnitude',
        magnitude_range=(
            0,
            0.45,
        ),
        type='Translate'),
]
resume_from = None
runner = dict(max_epochs=300, type='EpochBasedRunner')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        size=(
            256,
            -1,
        ),
        type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(keys=[
        'img',
    ], type='ImageToTensor'),
    dict(keys=[
        'img',
    ], type='Collect'),
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        size=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', flip_prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        magnitude_level=9,
        magnitude_std=0.5,
        num_policies=2,
        policies=[
            dict(type='AutoContrast'),
            dict(type='Equalize'),
            dict(type='Invert'),
            dict(
                magnitude_key='angle',
                magnitude_range=(
                    0,
                    30,
                ),
                type='Rotate'),
            dict(
                magnitude_key='bits',
                magnitude_range=(
                    4,
                    0,
                ),
                type='Posterize'),
            dict(
                magnitude_key='thr',
                magnitude_range=(
                    256,
                    0,
                ),
                type='Solarize'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    110,
                ),
                type='SolarizeAdd'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.9,
                ),
                type='ColorTransform'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.9,
                ),
                type='Contrast'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.9,
                ),
                type='Brightness'),
            dict(
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.9,
                ),
                type='Sharpness'),
            dict(
                direction='horizontal',
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.3,
                ),
                type='Shear'),
            dict(
                direction='vertical',
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.3,
                ),
                type='Shear'),
            dict(
                direction='horizontal',
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.45,
                ),
                type='Translate'),
            dict(
                direction='vertical',
                magnitude_key='magnitude',
                magnitude_range=(
                    0,
                    0.45,
                ),
                type='Translate'),
        ],
        total_level=10,
        type='RandAugment'),
    dict(
        erase_prob=0.25,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='Normalize'),
    dict(keys=[
        'img',
    ], type='ImageToTensor'),
    dict(keys=[
        'gt_label',
    ], type='ToTensor'),
    dict(keys=[
        'img',
        'gt_label',
    ], type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
