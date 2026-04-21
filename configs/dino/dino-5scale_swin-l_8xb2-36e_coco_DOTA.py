_base_ = './dino-5scale_swin-l_8xb2-12e_coco.py'

dataset_type = 'CocoDataset'
data_root = '/data1/sjc/work2/mmdetection/data/DOTA_coco'
classes = ('airplane', 'ship', 'storage-tank',)

fp16 = dict(loss_scale='dynamic')

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data1/sjc/work2/mmdetection/data/DOTA_coco/annotations/instances_train2017.json',
        data_prefix=dict(img='/data1/sjc/work2/mmdetection/data/DOTA_coco/train2017/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='/data1/sjc/work2/mmdetection/data/DOTA_coco/annotations/instances_val2017.json',
        data_prefix=dict(img='/data1/sjc/work2/mmdetection/data/DOTA_coco/val2017/')))
val_evaluator = dict(
    ann_file='/data1/sjc/work2/mmdetection/data/DOTA_coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

max_epochs = 20
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 12, 16],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=2)

work_dir = './work_dirs/dino_test/DO'
