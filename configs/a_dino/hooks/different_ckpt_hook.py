# configs/a_dino/hooks/load_backbone_from_a_hook.py

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class LoadBackboneFromAHook(Hook):
    def __init__(self, backbone_ckpt: str, other_ckpt: str):
        self.backbone_ckpt = backbone_ckpt
        self.other_ckpt = other_ckpt

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Step 1: 加载 backbone 权重
        print(f'🔁 正在加载 backbone 权重 from {self.backbone_ckpt}')
        backbone_state_dict = torch.load(self.backbone_ckpt, map_location='cpu')['state_dict']
        backbone_weights = {
            k.replace('backbone.', ''): v
            for k, v in backbone_state_dict.items()
            if k.startswith('backbone.')
        }
        model.backbone.load_state_dict(backbone_weights, strict=False)
        print('✅ backbone 权重加载完成')

        # Step 2: 加载其余权重
        print(f'🔁 正在加载其余权重 from {self.other_ckpt}')
        other_state_dict = torch.load(self.other_ckpt, map_location='cpu')['state_dict']
        for name, param in other_state_dict.items():
            if not name.startswith('backbone.'):
                try:
                    model.state_dict()[name].copy_(param)
                except Exception as e:
                    print(f'⚠️ 跳过加载参数: {name} - {e}')
        print('✅ 其余部分权重加载完成')
        print(f'✅ 成功从 {self.backbone_ckpt} 加载 backbone 权重，共 {len(backbone_weights)} 个参数')
