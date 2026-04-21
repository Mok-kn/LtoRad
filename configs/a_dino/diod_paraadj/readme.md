work3/mmdetection/configs/a_dino/diod_paraadj中主要调整了一些训练超参数参数
按照https://github.com/open-mmlab/mmdetection/tree/main/configs/dino进行了调整

除了backbone的冻结层数不变，具体为：
model = dict(
    backbone=dict(frozen_stages=1),
    bbox_head=dict(loss_cls=dict(loss_weight=2.0)),
    positional_encoding=dict(offset=-0.5, temperature=10000),
    dn_cfg=dict(group_cfg=dict(num_dn_queries=300)))

optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))